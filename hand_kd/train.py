"""Full 38-epoch paired-view A/B training with one frozen MobRecon teacher."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from hand_kd.data import ROOT, CombinedDataset, PilotDataset, PlannedDataset, load_selection
from hand_kd.evaluation import evaluate
from hand_kd.losses import paired_objective, teacher_targets
from hand_kd.model import load_mobrecon, setup_torch, SingleStackMobRecon, load_single_stack, verify_reload
from hand_kd.utils import epoch_indices, cpu_copy, atomic_save, load_bundle, tensor_digest, write_json, replace_json, seed_worker, verify, sha256_file

ARMS = ('A', 'B')


def load_training_config(args, root=ROOT):
    """Combine a shared parameter recipe with local assets supplied on the CLI."""
    config = json.loads(args.config.read_text())
    keys = {'seed', 'epochs', 'batch_size', 'workers', 'lr', 'crop_base_scale',
            'feature_weight', 'checkpoint_interval_batches'}
    if set(config) != keys:
        raise ValueError(f'Training recipe must contain only: {sorted(keys)}')
    for key in ('seed', 'epochs', 'batch_size', 'workers', 'checkpoint_interval_batches'):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f'Invalid integer parameter: {key}')
    for key in ('lr', 'crop_base_scale', 'feature_weight'):
        if not isinstance(config[key], (int, float)) or not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'Expected a positive finite {key}')
    if config['epochs'] != 38 or config['batch_size'] != 32 or not config['checkpoint_interval_batches']:
        raise ValueError('This trainer uses the 38-epoch, batch32 recipe')
    for key, argument in [('dataset_manifest', args.dataset_manifest),
                          ('dataset_selection', args.selection), ('teacher_checkpoint', args.teacher)]:
        path = (root / argument).resolve()
        config[key], config[key + '_sha256'] = str(path), sha256_file(path)
    config['upstream_revision'] = json.loads(Path(config['dataset_manifest']).read_text())['upstream_revision']
    manifest = (root / args.validation_manifest).resolve()
    cache = (root / args.validation_cache).resolve()
    index = json.loads((cache / 'index.json').read_text())
    template = root / 'external/mobrecon/template'
    config['validation'] = dict(manifest=str(manifest), manifest_sha256=sha256_file(manifest),
        cache=str(cache), cache_index_sha256=sha256_file(cache / 'index.json'),
        regressor_sha256=index['regressor_sha256'],
        topology_sha256=sha256_file(template / 'right_faces.npy'),
        transform_sha256=sha256_file(template / 'transform.pkl'))
    return config


def epoch_plan(admitted,seed,epoch,batch_size):
    order=epoch_indices(admitted,seed,epoch)
    count=len(order)//batch_size*batch_size
    return order[:count],order[count:]


def verify_inputs(config):
    manifest_path=Path(config['dataset_manifest'])
    assets={str(manifest_path):config['dataset_manifest_sha256'],
            config['dataset_selection']:config['dataset_selection_sha256'],
            config['teacher_checkpoint']:config['teacher_checkpoint_sha256']}
    manifest=json.loads(manifest_path.read_text())
    assets.update({str(manifest_path.parent/k):v for k,v in manifest['files'].items()})
    assets.update({v['path']:v['sha256'] for v in manifest['freihand']['arrays'].values()})
    assets[manifest['freihand']['archive']]=manifest['freihand']['sha256']
    assets.update({v['path']:v['sha256'] for v in manifest['comphand']['archives'].values()})
    prior=config['validation']
    assets[str(ROOT/prior['manifest'])]=prior['manifest_sha256']
    assets[str(ROOT/prior['cache']/'index.json')]=prior['cache_index_sha256']
    for name,key in [('j_reg.npy','regressor_sha256'),('right_faces.npy','topology_sha256'),('transform.pkl','transform_sha256')]:
        assets[str(ROOT/'external/mobrecon/template'/name)]=prior[key]
    for path,digest in assets.items():
        print(json.dumps(dict(checking=path)),flush=True);verify(Path(path),digest)
    revision=subprocess.check_output(['git','-C',str(ROOT/'external/mobrecon'),'rev-parse','HEAD'],text=True).strip()
    if revision!=config['upstream_revision'] or revision!=manifest['upstream_revision']:
        raise ValueError('Upstream revision changed')
    return assets,prior


def save_checkpoint(out,state,models,optimizers):
    state['arms']={a:dict(model=cpu_copy(models[a].state_dict()),optimizer=cpu_copy(optimizers[a].state_dict())) for a in ARMS}
    state.update(torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                 numpy_rng=np.random.get_state(),python_rng=random.getstate())
    atomic_save(out/'latest.pt',state)
    del state['arms']


def finalize(out,config):
    rows=[json.loads((out/f'epoch-{i:03d}'/'result.json').read_text()) for i in range(config['epochs'])]
    final={a:rows[-1]['arms'][a]['metrics'] for a in ARMS}
    result=dict(complete=True,epochs=config['epochs'],updates_per_arm=7422*config['epochs'],
        images_per_arm=237504*config['epochs'],augmented_views_per_arm=2*237504*config['epochs'],
        final=final,delta_B_minus_A_mpjpe_mm=final['B']['mpjpe_mm']-final['A']['mpjpe_mm'],
        best_development_epoch={a:min(rows,key=lambda r:r['arms'][a]['metrics']['mpjpe_mm'])['epoch'] for a in ARMS},
        primary='Fixed final epoch38; best development epochs are descriptive only',
        teacher=json.loads((out/'teacher-validation.json').read_text()),
        initial=json.loads((out/'initial-validation.json').read_text()),candidate_promoted=False,
        limitations='One seed; reused in-domain 256-image development set; no phone or deployment claim')
    replace_json(out/'curve.json',rows)
    replace_json(out/'complete.json',result)
    replace_json(out/'status.json',dict(stage='complete',completed_epochs=config['epochs'],candidate_promoted=False))
    print(json.dumps(result),flush=True)


def run(args):
    config=load_training_config(args)
    if args.check_inputs:
        verify_inputs(config)
        return
    out=args.output
    if args.resume:
        if not out.is_dir():raise ValueError('Missing resume directory')
    else:out.mkdir(parents=True,exist_ok=False)
    lock=(out/'writer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'STOP').exists():
        replace_json(out/'status.json',dict(stage='stop_retained',smoke=args.smoke))
        lock.close()
        return
    source_paths=(list((ROOT/'hand_kd').glob('*.py'))
                  + list((ROOT/'experiments/single_stack').glob('*.py'))
                  + [ROOT/'experiments/__init__.py']
                  + list((ROOT/'external/mobrecon').rglob('*.py')))
    sources={str(p.relative_to(ROOT)):sha256_file(p) for p in sorted(set(source_paths))}
    sources['configs/train.json']=sha256_file(args.config)
    assets,prior=verify_inputs(config)
    setup_torch(config['seed']);random.seed(config['seed'])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    base=CombinedDataset(config['dataset_manifest'],crop_base_scale=config['crop_base_scale'])
    admitted=load_selection(base,config['dataset_selection'],config['dataset_selection_sha256']).indices
    if len(admitted)!=237517:raise ValueError('Full clean training denominator changed')
    for name in ('plans','logs','interruptions'): (out/name).mkdir(exist_ok=True)
    if args.resume:
        state=load_bundle(out/'latest.pt',config,sources)
        if state['assets']!=assets or state['smoke']!=args.smoke:raise ValueError('Resume identity mismatch')
        for name,digest in sources.items():verify(out/'source-snapshot'/name,digest)
        for p in (out/'logs').glob('*.jsonl'):
            if str(p.relative_to(out)) not in state['logs']:
                p.rename(out/'interruptions'/f'{time.time_ns()}-{p.name}')
        partial=out/f"epoch-{state['next_epoch']:03d}"
        if partial.exists():partial.rename(out/'interruptions'/f'{time.time_ns()}-{partial.name}')
    else:
        for name in sources:
            target=out/'source-snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(args.config if name=='configs/train.json' else ROOT/name,target)
        write_json(out/'run.json',dict(config=config,sources=sources,assets=assets,smoke=args.smoke,
                   samples=len(admitted),batches_per_epoch=7422,images_per_epoch=237504,
                   command=sys.argv,torch=torch.__version__,gpu=torch.cuda.get_device_name(),
                   paired_views=True,online_teacher='same exact augmented 128x128 student tensors, FP32'))
        write_json(out/'config.json',config)
        (out/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
        state=dict(config=config,sources=sources,assets=assets,smoke=args.smoke,next_epoch=0,next_batch=0,
                   global_batches=0,logs=[],epoch_totals={},batch_chain='',training_seconds=0.)
    for epoch in range(38):
        plan,dropped=epoch_plan(admitted,config['seed'],epoch,config['batch_size'])
        path=out/'plans'/f'epoch-{epoch:03d}.npy'
        if args.resume:np.testing.assert_array_equal(np.load(path),plan)
        else:
            np.save(path,plan)
            write_json(path.with_suffix('.json'),dict(offered=len(admitted),seen=len(plan),
                batches=7422,dropped_original_indices=dropped.tolist(),sha256=sha256_file(path),
                augmentation_seed=config['seed']+epoch*len(admitted)))
    from utils.warmup_scheduler import adjust_learning_rate
    teacher=load_mobrecon(ROOT,Path(config['teacher_checkpoint'])).eval().requires_grad_(False)
    teacher_digest=tensor_digest(teacher.state_dict())
    models={a:SingleStackMobRecon(teacher).cuda().requires_grad_(True) for a in ARMS}
    optimizers={a:torch.optim.Adam(models[a].parameters(),lr=config['lr'],weight_decay=0) for a in ARMS}
    validation=PilotDataset(ROOT/prior['manifest'],ROOT/prior['cache'],ROOT/'external/mobrecon/template/j_reg.npy','validation')
    validation_loader=DataLoader(validation,batch_size=16,num_workers=0)
    reg=torch.from_numpy(np.load(ROOT/'external/mobrecon/template/j_reg.npy').astype(np.float32)).cuda()
    faces=torch.from_numpy(np.load(ROOT/'external/mobrecon/template/right_faces.npy').astype(np.int64)).cuda()
    if args.resume:
        for a in ARMS:
            models[a].load_state_dict(state['arms'][a]['model'],strict=True);optimizers[a].load_state_dict(state['arms'][a]['optimizer'])
        del state['arms']
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
    else:
        initial=tensor_digest(models['A'].state_dict())
        assert initial==tensor_digest(models['B'].state_dict())
        torch.save(cpu_copy(models['A'].state_dict()),out/'initial-student.pt')
        write_json(out/'initialization.json',dict(state_sha256=initial,parameters=sum(p.numel() for p in models['A'].parameters()),
                   source='retained layers copied from frozen teacher; fresh optimizers, no pilot continuation'))
        t=evaluate(teacher,validation_loader,reg,'cuda',out/'teacher-validation')
        if abs(t['mpjpe_mm']-14.539)>.003:raise ValueError('Teacher reference did not reproduce')
        evaluate(models['A'],validation_loader,reg,'cuda',out/'initial-validation')
        save_checkpoint(out,state,models,optimizers)
    if state['next_epoch']>=38:finalize(out,config);return
    stopped=[False]
    signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True))
    signal.signal(signal.SIGINT,lambda *_:stopped.__setitem__(0,True))
    session_start=time.monotonic();session_batches=0;torch.cuda.reset_peak_memory_stats()
    for epoch in range(state['next_epoch'],38):
        if stopped[0] or (out/'STOP').exists():
            save_checkpoint(out,state,models,optimizers)
            replace_json(out/'status.json',dict(stage='paused',next_epoch=state['next_epoch'],
                next_batch=state['next_batch'],global_batches=state['global_batches'],smoke=args.smoke))
            return
        start_batch=state['next_batch'];plan=np.load(out/'plans'/f'epoch-{epoch:03d}.npy')
        data=PlannedDataset(base,plan,config['seed']+epoch*len(admitted))
        remaining=Subset(data,range(start_batch*32,len(plan)))
        loader=DataLoader(remaining,batch_size=32,shuffle=False,drop_last=True,num_workers=config['workers'],
                          pin_memory=True,worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(config['seed']+epoch))
        for model in models.values():model.train()
        log=None
        for offset,batch in enumerate(loader):
            if stopped[0] or (out/'STOP').exists():
                if log is not None:
                    log.close();state['logs'].append(relative)
                save_checkpoint(out,state,models,optimizers)
                replace_json(out/'status.json',dict(stage='paused',next_epoch=state['next_epoch'],
                    next_batch=state['next_batch'],global_batches=state['global_batches'],smoke=args.smoke))
                return
            step=start_batch+offset;tick=time.monotonic()
            expected=plan[step*32:(step+1)*32]
            np.testing.assert_array_equal(batch['sample_index'].numpy(),expected)
            batch_digest=tensor_digest({k:v for k,v in batch.items() if isinstance(v,torch.Tensor)})
            gpu={k:v.cuda(non_blocking=True) for k,v in batch.items()}
            target=teacher_targets(teacher,gpu['img'])
            target_digest=tensor_digest({'feature':target})
            measures={}
            # Smoke spans epoch0 and consistency-enabled epoch5 without claiming either complete.
            effective_epoch=(0 if state['global_batches']<16 else 5) if args.smoke else epoch
            for arm in ARMS:
                optimizer=optimizers[arm];optimizer.zero_grad(set_to_none=True)
                adjust_learning_rate(optimizer,effective_epoch,step,7422,config['lr'],.1,[30],0)
                losses=paired_objective(models[arm],gpu,target,faces,effective_epoch,config['feature_weight'] if arm=='B' else 0.)
                if not torch.isfinite(losses['total']):raise FloatingPointError(f'Nonfinite objective in {arm}')
                losses['total'].backward()
                grad=torch.nn.utils.clip_grad_norm_(models[arm].parameters(),float('inf'),error_if_nonfinite=True)
                if grad<=0:raise FloatingPointError('Missing gradient')
                optimizer.step()
                measures[arm]={k:float(v.detach()) for k,v in losses.items()}
                measures[arm]['gradient_norm']=float(grad)
                for key,value in measures[arm].items():
                    sums=state['epoch_totals'].setdefault(arm,{})
                    sums[key]=sums.get(key,0.)+value
                del losses
            state['global_batches']+=1;state['next_batch']=step+1;session_batches+=1
            state['batch_chain']=hashlib.sha256((state['batch_chain']+batch_digest+target_digest).encode()).hexdigest()
            state['training_seconds']+=time.monotonic()-tick
            if log is None:
                relative=f'logs/epoch-{epoch:03d}-from-{step:05d}.jsonl';log=(out/relative).open('x')
            row=dict(epoch=epoch+1,effective_loss_epoch=effective_epoch,batch=step+1,global_batch=state['global_batches'],
                     original_indices=expected.tolist(),batch_sha256=batch_digest,teacher_feature_sha256=target_digest,
                     arms=measures,lr=optimizers['A'].param_groups[0]['lr'],seconds=time.monotonic()-tick)
            log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
            if step==start_batch:
                proof=out/f"audit-epoch-{epoch:03d}-batch-{step:05d}-{time.time_ns()}.pt"
                torch.save(dict(batch=batch,teacher_features=target.cpu()),proof)
            stop=stopped[0] or (out/'STOP').exists() or (args.smoke and args.stop_after_batches and session_batches>=args.stop_after_batches)
            checkpoint_due=(step+1)%config['checkpoint_interval_batches']==0 or step+1==7422 or stop
            if checkpoint_due:
                log.close();log=None;state['logs'].append(relative);save_checkpoint(out,state,models,optimizers)
            if (step+1)%25==0 or stop or step==start_batch:
                elapsed=time.monotonic()-session_start
                status=dict(stage='training',epoch=epoch+1,batch=step+1,batches_per_epoch=7422,
                    completed_epochs=epoch,global_batches=state['global_batches'],updates_per_arm=state['global_batches'],
                    epochs=38,full_pool=237517,images_per_epoch=237504,lr=optimizers['A'].param_groups[0]['lr'],
                    session_seconds=elapsed,session_batches=session_batches,last_losses=measures,
                    estimated_remaining_hours=elapsed/session_batches*(38*7422-state['global_batches'])/3600,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),smoke=args.smoke)
                replace_json(out/'status.json',status);print(json.dumps(status),flush=True)
            if stop:
                smoke_result={}
                if args.smoke:
                    for arm in ARMS:
                        sample=next(iter(validation_loader))['image_student'][:2].cuda()
                        path=out/f"smoke-{arm}-{state['global_batches']:05d}.pt"
                        torch.save(cpu_copy(models[arm].state_dict()),path)
                        loaded=load_single_stack(ROOT,path).eval()
                        parity=verify_reload(models[arm],loaded,sample);del loaded
                        metrics=evaluate(models[arm],validation_loader,reg,'cuda',out/f"smoke-{arm}-{state['global_batches']:05d}-validation")
                        smoke_result[arm]=dict(reload=parity,metrics=metrics)
                    assert tensor_digest(teacher.state_dict())==teacher_digest
                    write_json(out/f"smoke-result-{state['global_batches']:05d}.json",dict(arms=smoke_result,
                        steps=state['global_batches'],batch_chain=state['batch_chain'],teacher_unchanged=True,
                        seconds=time.monotonic()-session_start,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
                replace_json(out/'status.json',dict(stage='smoke_paused' if args.smoke else 'paused',next_epoch=epoch,
                    next_batch=step+1,global_batches=state['global_batches'],smoke=args.smoke))
                return
            del gpu,batch,target
        assert tensor_digest(teacher.state_dict())==teacher_digest
        folder=out/f'epoch-{epoch:03d}';folder.mkdir(exist_ok=False)
        results={}
        for arm in ARMS:
            metrics=evaluate(models[arm],validation_loader,reg,'cuda',folder/arm)
            path=folder/f'{arm}-student.pt';torch.save(cpu_copy(models[arm].state_dict()),path)
            results[arm]=dict(metrics=metrics,checkpoint_sha256=sha256_file(path),
                             average_losses={k:v/7422 for k,v in state['epoch_totals'][arm].items()})
        write_json(folder/'result.json',dict(epoch=epoch+1,updates_per_arm=state['global_batches'],
                   images_per_arm=237504,arms=results,batch_chain=state['batch_chain'],teacher_unchanged=True))
        state.update(next_epoch=epoch+1,next_batch=0,epoch_totals={})
        save_checkpoint(out,state,models,optimizers)
        replace_json(out/'status.json',dict(stage='epoch_complete',completed_epochs=epoch+1,epochs=38,arms=results))
    finalize(out,config)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/train.json')
    parser.add_argument('--dataset-manifest',type=Path,required=True)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--teacher',type=Path,required=True)
    parser.add_argument('--validation-manifest',type=Path,required=True)
    parser.add_argument('--validation-cache',type=Path,required=True)
    parser.add_argument('--check-inputs',action='store_true',help='Verify asset hashes without starting training')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--stop-after-batches',type=int)
    args=parser.parse_args()
    if not args.check_inputs and args.output is None:parser.error('--output is required for training')
    if args.stop_after_batches and not args.smoke:parser.error('Bounded stop is smoke-only')
    try:run(args)
    except Exception:
        if args.output is not None and args.output.is_dir():
            path=args.output/f'failure-{time.time_ns()}.txt';path.write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
