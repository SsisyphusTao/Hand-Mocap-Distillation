"""Historical three-arm, ten-epoch WiLoR transfer protocol.

The recorded run was stopped by the user after 2.31 epochs. New invocations
create new experiments and cannot resume the original source-bound checkpoint.
"""
import argparse
import fcntl
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from hand_kd.data import ROOT, CombinedDataset, PilotDataset, load_selection
from hand_kd.evaluation import evaluate
from hand_kd.model import load_mobrecon, setup_torch
from hand_kd.utils import write_json, replace_json, verify, seed_worker, sha256_file, epoch_indices, cpu_copy, atomic_save, load_bundle
from experiments.wilor_transfer.features import StudentTaps, teacher_input
from experiments.wilor_transfer.support import load_wilor, training_mode, DualPlan, tensor_digest, source_files, verify_assets, make_head, comparison_gate, backward_batch, verify_stream_assets

ARMS = ('supervised', 'fresh', 'prefitted')


def checkpoint(out,state,models,heads,optimizers):
    state['arms']={arm:dict(model=cpu_copy(models[arm].state_dict()),
        head=cpu_copy(heads[arm].state_dict()) if heads[arm] else None,
        optimizer=cpu_copy(optimizers[arm].state_dict())) for arm in ARMS}
    state['torch_rng']=torch.get_rng_state();state['cuda_rng']=torch.cuda.get_rng_state_all()
    state['python_rng']=random.getstate();state['numpy_rng']=np.random.get_state()
    atomic_save(out/'latest.pt',state)
    # CPU copies must not survive in RAM between checkpoints.
    del state['arms']


def teacher_targets(teacher,cfg,rgb,batch_size):
    values=[]
    with torch.no_grad():
        for view in range(2):
            parts=[]
            for chunk in rgb[:,view*3:view*3+3].split(batch_size):
                feature=teacher.backbone(teacher_input(chunk.cuda(),cfg.MODEL.IMAGE_MEAN,cfg.MODEL.IMAGE_STD))[-1]
                if feature.shape[1:]!=(1280,16,12) or not torch.isfinite(feature).all():
                    raise ValueError('Teacher feature contract failed')
                parts.append(F.adaptive_avg_pool2d(feature,(8,6)).half())
            values.append(torch.cat(parts))
    return torch.stack(values,1)


def build_sources(config_path,cfg,old):
    paths=source_files()+[config_path.resolve(),ROOT/cfg['parent_config'],ROOT/cfg['prefit_config']]
    return {str(p.relative_to(ROOT)):sha256_file(p) for p in sorted(set(paths))}


def finalize_run(out,cfg,images):
    # Rebuild after a crash between the last shared checkpoint and reporting.
    curve=[json.loads((out/f'epoch-{i:03d}'/'result.json').read_text()) for i in range(cfg['epochs'])]
    replace_json(out/'curve.json',curve)
    baseline=json.loads((out/'baseline.json').read_text());last=curve[-1]['arms']
    best={a:min(curve,key=lambda r:r['arms'][a]['metrics']['mpjpe_mm'])['epoch'] for a in ARMS}
    replace_json(out/'comparison.json',dict(completed_epochs=cfg['epochs'],images_per_epoch=images,
        images_per_arm=images*cfg['epochs'],final=last,best_development_epoch=best,
        final_prefit_gate=comparison_gate(last['prefitted']['metrics'],last['supervised']['metrics'],last['fresh']['metrics'],baseline),
        candidate_promoted=False,smoke=images!=237517,scope='additional full epochs; reused development set, not independent test'))
    replace_json(out/'status.json',dict(stage='complete',completed_epochs=cfg['epochs'],candidate_promoted=False,smoke=images!=237517))


def run(args):
    cfg=json.loads(args.config.read_text())
    if args.smoke:cfg={**cfg,'epochs':2,'expected_images_per_epoch':45,'checkpoint_interval_batches':1,'workers':0}
    old=json.loads((ROOT/cfg['parent_config']).read_text())
    prefit_cfg=json.loads((ROOT/cfg['prefit_config']).read_text())
    out=args.output
    if args.resume:
        if not out.is_dir():raise ValueError('Missing resume directory')
    else:out.mkdir(parents=True,exist_ok=False)
    lock=(out/'writer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sources=build_sources(args.config,cfg,old)
    verify(Path(old['initial_checkpoint']),old['initial_checkpoint_sha256'])
    verify(ROOT/old['teacher_checkpoint'],old['teacher_checkpoint_sha256'])
    prefit=Path(prefit_cfg['prefit_run'])/'frozen-head.pt';verify(prefit,prefit_cfg['head_sha256'])
    if args.resume:
        state=load_bundle(out/'latest.pt',cfg,sources)
        run_doc=json.loads((out/'run.json').read_text())
        for name,digest in sources.items():verify(out/'source-snapshot'/name,digest)
        if run_doc['assets']!=state['assets']:raise ValueError('Resume asset identity mismatch')
        verify_stream_assets(state['assets'])
    else:
        assets,revisions=verify_assets(old)
        for name in sources:
            target=out/'source-snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,target)
        write_json(out/'config.json',cfg)
        write_json(out/'run.json',dict(config=cfg,sources=sources,assets=assets,revisions=revisions,
            command=sys.argv,smoke=args.smoke,torch=torch.__version__,cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),initial_checkpoint_sha256=old['initial_checkpoint_sha256'],
            prefit_head_sha256=prefit_cfg['head_sha256'],validation_scope=old['validation_scope']))
        state=dict(config=cfg,sources=sources,assets=assets,next_epoch=0,next_batch=0,logs=[],totals={},images=0,
                   batch_chain='',global_batches=0)
    base=CombinedDataset(old['dataset_manifest'],crop_base_scale=1.5)
    admitted=load_selection(base,old['dataset_selection'],old['dataset_selection_sha256']).indices
    if args.smoke:admitted=admitted[np.random.default_rng(cfg['seed']).permutation(len(admitted))[:45]]
    if len(admitted)!=cfg['expected_images_per_epoch']:raise ValueError('Epoch denominator mismatch')
    (out/'plans').mkdir(exist_ok=True);(out/'logs').mkdir(exist_ok=True);(out/'interruptions').mkdir(exist_ok=True)
    for epoch in range(cfg['epochs']):
        plan=epoch_indices(admitted,cfg['seed'],epoch);path=out/'plans'/f'epoch-{epoch:03d}.npy'
        if args.resume:np.testing.assert_array_equal(np.load(path),plan)
        else:
            np.save(path,plan)
            write_json(path.with_suffix('.json'),dict(images=len(plan),unique=len(np.unique(plan)),sha256=sha256_file(path),
                batches=math.ceil(len(plan)/cfg['batch_size']),augmentation_seed=cfg['seed']+epoch*len(plan)))
    # Uncommitted logs survive, but are not counted as resumed training.
    for path in (out/'logs').glob('*.jsonl'):
        if str(path.relative_to(out)) not in state['logs']:
            path.rename(out/'interruptions'/f'{time.time_ns()}-{path.name}')
    prior=old['validation']
    validation=PilotDataset(ROOT/prior['manifest'],ROOT/prior['cache'],ROOT/'external/mobrecon/template/j_reg.npy','validation')
    validation_loader=DataLoader(validation,batch_size=16,shuffle=False,num_workers=0)
    reg=torch.from_numpy(np.load(ROOT/'external/mobrecon/template/j_reg.npy').astype(np.float32)).cuda()
    faces=torch.from_numpy(np.load(ROOT/'external/mobrecon/template/right_faces.npy').astype(np.int64)).cuda()
    models={};heads={};taps={};optimizers={};buffers={}
    for arm in ARMS:
        setup_torch(cfg['seed']);random.seed(cfg['seed'])
        model=load_mobrecon(ROOT,Path(old['initial_checkpoint']));models[arm]=model
        buffers[arm]={k:v.detach().cpu().clone() for k,v in model.named_buffers()}
        head=make_head(arm,cfg['seed']+1,prefit);heads[arm]=head
        optimizer=torch.optim.Adam(list(model.parameters())+(list(head.parameters()) if head else []),
                                   lr=cfg['lr'],weight_decay=cfg['weight_decay']);optimizers[arm]=optimizer
        if args.resume:
            saved=state['arms'][arm];model.load_state_dict(saved['model'],strict=True)
            if head:head.load_state_dict(saved['head'],strict=True)
            optimizer.load_state_dict(saved['optimizer'])
        taps[arm]=StudentTaps(model) if head else None
    if args.resume:
        del state['arms']
    else:
        initial=tensor_digest(models['supervised'].state_dict())
        assert all(tensor_digest(m.state_dict())==initial for m in models.values())
        baseline=evaluate(models['supervised'],validation_loader,reg,'cuda',out/'baseline')
        write_json(out/'initialization.json',dict(student_digest=initial,
            heads={a:tensor_digest(h.state_dict()) if h else None for a,h in heads.items()},baseline=baseline))
        checkpoint(out,state,models,heads,optimizers)
    if state['next_epoch']>=cfg['epochs']:
        finalize_run(out,cfg,len(admitted))
        print('All configured epochs already complete; reports verified.',flush=True);return
    # All models fit concurrently with sample-weighted student microbatches.
    teacher,teacher_cfg=load_wilor(ROOT);teacher_before=tensor_digest(teacher.state_dict())
    assert all(not p.requires_grad for p in teacher.parameters())
    if args.resume:
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
    session_start=time.monotonic();session_batches=0;torch.cuda.reset_peak_memory_stats()
    n_batches=math.ceil(len(admitted)/cfg['batch_size'])
    print(json.dumps(dict(stage='ready',epochs=cfg['epochs'],images_per_epoch=len(admitted),
                         batches_per_epoch=n_batches,resume_epoch=state['next_epoch'],resume_batch=state['next_batch'])),flush=True)
    for epoch in range(state['next_epoch'],cfg['epochs']):
        start_batch=state['next_batch'];plan=np.load(out/'plans'/f'epoch-{epoch:03d}.npy')
        dual=DualPlan(base,plan,cfg['seed']+epoch*len(plan))
        dataset=Subset(dual,range(start_batch*cfg['batch_size'],len(plan)))
        loader=DataLoader(dataset,batch_size=cfg['batch_size'],shuffle=False,drop_last=False,
            num_workers=cfg['workers'],worker_init_fn=seed_worker,pin_memory=True,
            generator=torch.Generator().manual_seed(cfg['seed']+epoch))
        for arm in ARMS:
            training_mode(models[arm],'released_stats')
            if heads[arm]:heads[arm].train()
        log=None;log_relative=None
        for offset,batch in enumerate(loader):
            step=start_batch+offset;tick=time.monotonic();rgb=batch.pop('teacher_rgb')
            expected=plan[step*cfg['batch_size']:(step+1)*cfg['batch_size']]
            np.testing.assert_array_equal(batch['original_sample_index'].numpy(),expected)
            crop_error=float((rgb[:,:,::2,::2]-(batch['img']+1)/2).abs().max())
            if crop_error>2e-6:raise ValueError('Teacher/student crop mismatch')
            target=teacher_targets(teacher,teacher_cfg,rgb,cfg['teacher_batch_size'])
            target_digest=tensor_digest({'target':target});batch_digest=tensor_digest(
                {k:v for k,v in batch.items() if isinstance(v,torch.Tensor)})
            measures={}
            for arm in ARMS:
                optimizer=optimizers[arm];optimizer.zero_grad(set_to_none=True)
                losses=backward_batch(models[arm],heads[arm],taps[arm],batch,target,faces,
                                      cfg['microbatch_size'],cfg['feature_weight'])
                grad=torch.nn.utils.clip_grad_norm_(models[arm].parameters(),float('inf'),error_if_nonfinite=True)
                hgrad=torch.nn.utils.clip_grad_norm_(heads[arm].parameters(),float('inf'),error_if_nonfinite=True) if heads[arm] else None
                if grad<=0 or (hgrad is not None and hgrad<=0):raise FloatingPointError('Missing gradients')
                optimizer.step();measures[arm]=dict(**losses,student_gradient_norm=float(grad),
                    head_gradient_norm=float(hgrad) if hgrad is not None else None)
                sums=state['totals'].setdefault(arm,{})
                for key,value in losses.items():sums[key]=sums.get(key,0.)+value*len(rgb)
            if tensor_digest({'target':target})!=target_digest:raise ValueError('Teacher targets changed')
            state['images']+=len(rgb);state['global_batches']+=1;state['next_batch']=step+1;session_batches+=1
            state['batch_chain']=hashlib.sha256((state['batch_chain']+batch_digest+target_digest).encode()).hexdigest()
            if log is None:
                log_relative=f'logs/epoch-{epoch:03d}-from-{step:05d}.jsonl';log=(out/log_relative).open('x')
            row=dict(epoch=epoch+1,batch=step+1,images=len(rgb),original_indices=expected.tolist(),
                     rgb_sha256=batch['source_rgb_sha256'],batch_sha256=batch_digest,teacher_features_sha256=target_digest,
                     crop_max_abs=crop_error,arms=measures,seconds=time.monotonic()-tick)
            log.write(json.dumps(row)+'\n');log.flush()
            if step==0 or step+1==n_batches:
                proof=out/f'audit-epoch-{epoch:03d}-batch-{step:05d}.pt'
                if proof.exists():proof.rename(out/'interruptions'/f'{time.time_ns()}-{proof.name}')
                torch.save(dict(batch=batch,teacher_features=target.cpu()),proof)
            elapsed=time.monotonic()-session_start
            remaining=(cfg['epochs']-epoch-1)*n_batches+n_batches-step-1
            status=dict(stage='training',epoch=epoch+1,completed_batches=step+1,batches_per_epoch=n_batches,
                images_completed_in_epoch=state['images'],images_per_epoch=len(plan),epochs=cfg['epochs'],
                session_seconds=elapsed,session_batches=session_batches,
                estimated_remaining_hours=elapsed/session_batches*remaining/3600,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),last_batch_seconds=row['seconds'],
                committed_global_batches=state.get('committed_global_batches',0))
            replace_json(out/'status.json',status)
            if step%16==0 or step+1==n_batches:print(json.dumps(status),flush=True)
            stop=args.stop_after_batches is not None and session_batches>=args.stop_after_batches
            if (step+1)%cfg['checkpoint_interval_batches']==0 or step+1==n_batches or stop:
                log.close();log=None;state['logs'].append(log_relative)
                state['committed_global_batches']=state['global_batches'];checkpoint(out,state,models,heads,optimizers)
            if stop:
                if not args.smoke:raise ValueError('Stop-after is restricted to smoke tests')
                replace_json(out/'status.json',dict(stage='smoke_paused',epoch=epoch+1,next_batch=step+1));return
            del rgb,target,batch;gc.collect() if args.smoke else None
        if state['images']!=len(plan):raise ValueError('Incomplete full-epoch coverage')
        assert tensor_digest(teacher.state_dict())==teacher_before
        folder=out/f'epoch-{epoch:03d}'
        if folder.exists():folder.rename(out/'interruptions'/f'{time.time_ns()}-{folder.name}')
        folder.mkdir();results={}
        for arm in ARMS:
            for name,value in models[arm].named_buffers():assert torch.equal(value.cpu(),buffers[arm][name]),name
            metrics=evaluate(models[arm],validation_loader,reg,'cuda',folder/arm)
            torch.save(models[arm].state_dict(),folder/f'{arm}-student.pt')
            if heads[arm]:torch.save(heads[arm].state_dict(),folder/f'{arm}-head.pt')
            results[arm]=dict(metrics=metrics,checkpoint_sha256=sha256_file(folder/f'{arm}-student.pt'),
                             average_losses={k:v/len(plan) for k,v in state['totals'][arm].items()})
        write_json(folder/'result.json',dict(epoch=epoch+1,images=state['images'],batches=n_batches,arms=results,
            plan_sha256=sha256_file(out/'plans'/f'epoch-{epoch:03d}.npy'),batch_chain=state['batch_chain'],
            buffers_unchanged=True,teacher_unchanged=True))
        shutil.copyfile(out/'latest.pt',folder/'resume-at-epoch-end.pt')
        state.update(next_epoch=epoch+1,next_batch=0,totals={},images=0,batch_chain='')
        checkpoint(out,state,models,heads,optimizers)
        curve=[json.loads((out/f'epoch-{i:03d}'/'result.json').read_text()) for i in range(epoch+1)]
        replace_json(out/'curve.json',curve)
        print(json.dumps(dict(completed_epoch=epoch+1,metrics={a:results[a]['metrics'] for a in ARMS})),flush=True)
    finalize_run(out,cfg,len(admitted))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--stop-after-batches', type=int)
    args = parser.parse_args()
    if args.stop_after_batches is not None and not args.smoke:
        parser.error('--stop-after-batches is smoke-only')
    try:
        run(args)
    except Exception as error:
        if args.output.is_dir():
            write_json(args.output/f'failure-{time.time_ns()}.json',
                       dict(type=type(error).__name__,message=str(error),command=sys.argv))
        raise


if __name__ == '__main__':
    main()
