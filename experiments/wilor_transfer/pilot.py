"""Historical matched 256-step WiLoR-to-MobRecon feature pilot."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from hand_kd.data import ROOT, CombinedDataset, PilotDataset, load_selection
from hand_kd.evaluation import evaluate
from hand_kd.losses import paired_loss
from hand_kd.model import load_mobrecon, setup_torch
from hand_kd.utils import seed_worker, verify, write_json, replace_json, sha256_file
from experiments.wilor_transfer.features import StudentTaps, FeatureAligner, feature_loss, teacher_input, STUDENT_LAYERS
from experiments.wilor_transfer.support import load_wilor, training_mode, checked_batch, prediction_parity, pilot_gate, DualPlan, source_files, verify_assets, tensor_digest


def cache(config,output,smoke):
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'config.json',config)
    assets,revisions = verify_assets(config)
    snapshots = {}
    for path in source_files():
        relative = str(path.relative_to(ROOT)); snapshots[relative] = sha256_file(path)
        destination = output/'source-snapshot'/relative
        destination.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(path,destination)
    for name in revisions:
        diff = subprocess.check_output(['git','-C',str(ROOT/'external'/name),'diff','HEAD'],text=True)
        (output/f'{name}-upstream.diff').write_text(diff)
    write_json(output/'provenance.json',dict(sources=snapshots,assets=assets,revisions=revisions,
               command=sys.argv,torch=torch.__version__,gpu=torch.cuda.get_device_name(),
               smoke=smoke,config_sha256=sha256_file(output/'config.json')))
    with (output/'environment.txt').open('x') as stream:
        subprocess.run([sys.executable,'-m','pip','freeze'],stdout=stream,check=True)
    setup_torch(config['seed']); random.seed(config['seed'])
    base = CombinedDataset(config['dataset_manifest'],crop_base_scale=1.5)
    selected = load_selection(base,config['dataset_selection'],config['dataset_selection_sha256'])
    n = config['steps']*config['batch_size']
    slots = np.random.default_rng(config['seed']).permutation(len(selected))[:n]
    indices = selected.indices[slots]
    np.save(output/'training-indices.npy',indices)
    write_json(output/'sampling.json',dict(offered=len(base),admitted=len(selected),pilot=n,
               freihand=int((indices<base.fh_count).sum()),comphand=int((indices>=base.fh_count).sum()),
               seed=config['seed'],indices_sha256=sha256_file(output/'training-indices.npy'),
               scope='train partitions only; person/session independence and upstream pretraining exposure unverified'))
    data = DualPlan(base,indices,config['seed'])
    loader = DataLoader(data,batch_size=config['batch_size'],num_workers=config['workers'],
                        shuffle=False,worker_init_fn=seed_worker)
    teacher,cfg = load_wilor(ROOT)
    if any(p.requires_grad for p in teacher.parameters()): raise ValueError('Teacher is trainable')
    teacher_digest_before = tensor_digest(teacher.state_dict())
    teacher.eval()
    write_json(output/'teacher-config.json',{'effective_config':cfg.dump(),
               'feature':'backbone.forward last_norm image tokens, before refine_net',
               'input':'original RGB affine256, shared photometric draws, configured normalization, x32:224 trim',
               'pool':'adaptive average pool 16x12 to 8x6; cache float16; loss float32',
               'region_mapping':'normalized map extent; not identical CNN/ViT receptive-field centers'})
    (output/'cache').mkdir()
    records=[]; start=time.monotonic(); max_quant=0.; max_crop_error=0.
    torch.cuda.reset_peak_memory_stats()
    for step,batch in enumerate(loader):
        rgb = batch.pop('teacher_rgb')
        crop_error = float((rgb[:,:,::2,::2]-(batch['img']+1)/2).abs().max())
        max_crop_error=max(max_crop_error,crop_error)
        if crop_error>2e-6: raise ValueError(f'Crop correspondence failed: {crop_error}')
        rgb_views = torch.cat([rgb[:,:3],rgb[:,3:]],0)
        target=[]
        with torch.no_grad():
            for chunk in rgb_views.split(config['teacher_batch_size']):
                feature = teacher.backbone(teacher_input(chunk.cuda(),cfg.MODEL.IMAGE_MEAN,cfg.MODEL.IMAGE_STD))[-1]
                if feature.shape[1:] != (1280,16,12) or not torch.isfinite(feature).all():
                    raise ValueError('Teacher feature contract failed')
                pooled=F.adaptive_avg_pool2d(feature,(8,6)).cpu()
                packed=pooled.half()
                max_quant=max(max_quant,float((pooled-packed.float()).abs().max()))
                target.append(packed)
        batch['teacher_features'] = torch.stack(torch.cat(target).split(len(rgb)),1)
        if step==0:
            tiles=[]
            for i in range(min(4,len(rgb))):
                student=((batch['img'][i,:3].permute(1,2,0).numpy()+1)*127.5).clip(0,255).astype(np.uint8)
                high=(rgb[i,:3].permute(1,2,0).numpy()*255).clip(0,255).astype(np.uint8)
                high[:,31:33]=[255,0,0];high[:,223:225]=[255,0,0]
                tiles.append(np.concatenate([cv2.resize(student,(256,256)),high],1))
            cv2.imwrite(str(output/'crop-correspondence.png'),np.concatenate(tiles,0)[:,:,::-1])
            write_json(output/'feature-audit.json',dict(teacher_shape=[1280,16,12],cached_shape=list(batch['teacher_features'].shape),
                feature_std=float(batch['teacher_features'].float().std()),
                different_sample_l2=float((batch['teacher_features'][0].float()-batch['teacher_features'][1].float()).square().mean().sqrt()),
                crop_max_abs=crop_error,teacher_trainable_parameters=0))
        filename=f'cache/{step:04d}.pt'
        with (output/filename).open('xb') as stream: torch.save(batch,stream)
        records.append(dict(file=filename,sha256=sha256_file(output/filename),
                            slots=batch['slot'].tolist(),indices=batch['original_sample_index'].tolist()))
        replace_json(output/'status.json',dict(stage='cache',completed=step+1,total=len(loader),seconds=time.monotonic()-start))
        print(json.dumps(dict(cache=step+1,total=len(loader),seconds=round(time.monotonic()-start,2))),flush=True)
    if tensor_digest(teacher.state_dict()) != teacher_digest_before: raise ValueError('Teacher state changed')
    write_json(output/'cache-index.json',dict(records=records,crop_max_abs=max_crop_error,
               fp16_max_abs=max_quant,seconds=time.monotonic()-start,
               teacher_state_sha256=teacher_digest_before,teacher_unchanged=True,
               peak_allocated_bytes=torch.cuda.max_memory_allocated()))
    del teacher;gc.collect();torch.cuda.empty_cache()


def train(config,output):
    provenance=json.loads((output/'provenance.json').read_text())
    if json.loads((output/'config.json').read_text()) != config: raise ValueError('Config changed')
    for relative,digest in provenance['sources'].items(): verify(ROOT/relative,digest)
    verify(Path(config['initial_checkpoint']),config['initial_checkpoint_sha256'])
    for path,digest in provenance['assets'].items():
        # Archives are consumed only during cache generation. Recheck evaluation inputs.
        if not path.endswith('.zip'): verify(Path(path),digest)
    index=json.loads((output/'cache-index.json').read_text())
    records=index['records']
    if len(records)!=config['steps']: raise ValueError('Incomplete cache')
    prior=config['validation']
    validation=PilotDataset(ROOT/prior['manifest'],ROOT/prior['cache'],ROOT/'external/mobrecon/template/j_reg.npy','validation')
    loader=DataLoader(validation,batch_size=16,shuffle=False,num_workers=0)
    reg=torch.from_numpy(np.load(ROOT/'external/mobrecon/template/j_reg.npy').astype(np.float32)).cuda()
    faces=torch.from_numpy(np.load(ROOT/'external/mobrecon/template/right_faces.npy').astype(np.int64)).cuda()
    results={}; initial=None
    for arm in ('supervised','projected','region'):
        folder=output/arm;folder.mkdir(exist_ok=False)
        setup_torch(config['seed']);random.seed(config['seed'])
        model=load_mobrecon(ROOT,Path(config['initial_checkpoint']))
        bn_initial={k:v.detach().cpu().clone() for k,v in model.named_buffers()}
        metrics=evaluate(model,loader,reg,'cuda',folder/'initial')
        if initial is None: initial=metrics
        else:
            with np.load(output/'supervised/initial.npz') as reference, np.load(folder/'initial.npz') as current:
                np.testing.assert_array_equal(current['ids'],reference['ids'])
                np.testing.assert_array_equal(current['truth'],reference['truth'])
                np.testing.assert_allclose(current['prediction'],reference['prediction'],rtol=0,atol=2e-7)
        taps=StudentTaps(model) if arm!='supervised' else None
        torch.manual_seed(config['seed']+1)
        aligner=FeatureAligner(mode=arm).cuda() if taps else None
        optimizer=torch.optim.Adam(list(model.parameters())+(list(aligner.parameters()) if aligner else []),
                                   lr=config['lr'],weight_decay=0)
        training_mode(model,'released_stats')
        if aligner: aligner.train()
        digest=hashlib.sha256();start=time.monotonic();torch.cuda.reset_peak_memory_stats()
        with (folder/'steps.jsonl').open('x') as log:
            for step,record in enumerate(records):
                stored=checked_batch(output/record['file'],record['sha256'])
                digest.update(record['sha256'].encode())
                batch={k:v.cuda() for k,v in stored.items() if isinstance(v,torch.Tensor) and k!='teacher_features'}
                optimizer.zero_grad(set_to_none=True)
                predictions=[]; feature_losses=[]
                for view in range(2):
                    predictions.append(model(batch['img'][:,view*3:view*3+3]))
                    if aligner:
                        feature_losses.append(feature_loss(aligner(taps.features()),stored['teacher_features'][:,view].cuda()))
                        if step==0 and view==0:
                            write_json(folder/'student-features.json',dict(names=STUDENT_LAYERS,
                                shapes=[list(t.shape) for t in taps.features()],
                                auxiliary_parameters=sum(p.numel() for p in aligner.parameters())))
                        taps.clear()
                prediction=dict(verts=torch.cat([p['mesh_pred'] for p in predictions],-1),
                                joint_img=torch.cat([p['uv_pred'] for p in predictions],-1))
                losses=paired_loss(prediction,batch,faces,epoch=9)
                kd=torch.stack(feature_losses).mean() if feature_losses else losses['loss'].new_zeros(())
                total=losses['loss']+config['feature_weight']*kd
                if not torch.isfinite(total): raise ValueError('Nonfinite training loss')
                total.backward()
                grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
                if grad_norm<=0: raise ValueError('Missing student gradients')
                if aligner:
                    aux_norm=torch.nn.utils.clip_grad_norm_(aligner.parameters(),float('inf'),error_if_nonfinite=True)
                    if aux_norm<=0: raise ValueError('Missing auxiliary gradients')
                optimizer.step()
                row=dict(step=step+1,supervised=float(losses['loss'].detach()),feature=float(kd.detach()),
                         total=float(total.detach()),student_gradient_norm=float(grad_norm),
                         batch_sha256=record['sha256'])
                log.write(json.dumps(row)+'\n');log.flush()
                if step%16==0 or step+1==len(records):
                    replace_json(output/'status.json',dict(stage='train',arm=arm,completed=step+1,total=len(records),seconds=time.monotonic()-start))
                    print(json.dumps(dict(arm=arm,**row)),flush=True)
        train_seconds=time.monotonic()-start
        if taps: taps.close()
        for name,value in model.named_buffers():
            if not torch.equal(value.cpu(),bn_initial[name]): raise ValueError(f'Buffer changed: {name}')
        with (folder/'student.pt').open('xb') as stream: torch.save(model.state_dict(),stream)
        with (folder/'training-state.pt').open('xb') as stream:
            torch.save(dict(optimizer=optimizer.state_dict(),auxiliary=aligner.state_dict() if aligner else None,
                            step=len(records),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),stream)
        metrics=evaluate(model,loader,reg,'cuda',folder/'final')
        sample=checked_batch(output/records[0]['file'],records[0]['sha256'])['img'][:2,:3].cuda()
        with torch.no_grad(): expected=model(sample)
        reloaded=load_mobrecon(ROOT,folder/'student.pt').eval()
        for name,value in model.state_dict().items():
            if not torch.equal(value,reloaded.state_dict()[name]):
                raise ValueError(f'Reloaded parameter/buffer differs: {name}')
        with torch.no_grad(): actual=reloaded(sample)
        parity=prediction_parity(expected,actual)
        result=dict(metrics=metrics,checkpoint_sha256=sha256_file(folder/'student.pt'),
                    batch_digest=digest.hexdigest(),steps=len(records),seconds=train_seconds,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),buffers_unchanged=True,
                    student_only_reload_max_abs=parity,reload_tolerance=1e-6,checkpoint_tensors_exact=True,
                    student_parameters=sum(p.numel() for p in model.parameters()))
        write_json(folder/'result.json',result);results[arm]=result
        print(json.dumps(dict(completed=arm,**result)),flush=True)
        del model,reloaded,optimizer,aligner,predictions,prediction,losses,total,kd
        gc.collect();torch.cuda.empty_cache()
    if len({r['batch_digest'] for r in results.values()})!=1: raise ValueError('Batch exposure differs')
    gates={arm:pilot_gate(results[arm]['metrics'],results['supervised']['metrics'],initial) for arm in ('projected','region')}
    write_json(output/'comparison.json',dict(initial=initial,arms=results,gates=gates,
               cache_index_sha256=sha256_file(output/'cache-index.json'),smoke=provenance['smoke'],
               conclusion='Exploratory reused-development result; no independent test or visual acceptance'))
    replace_json(output/'status.json',dict(stage='complete',gates=gates))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=('cache','train','all'),default='all')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    if args.smoke:
        config={**config,'steps':4,'batch_size':4,'workers':0}
    try:
        if args.stage in ('cache','all'): cache(config,args.output,args.smoke)
        if args.stage in ('train','all'): train(config,args.output)
    except Exception as error:
        if args.output.is_dir():
            write_json(args.output/f'failure-{time.time_ns()}.json',dict(type=type(error).__name__,message=str(error),command=sys.argv))
        raise


if __name__ == '__main__':
    main()
