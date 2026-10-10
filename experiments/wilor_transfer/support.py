"""Shared contracts for the historical WiLoR feature-transfer runners."""
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

from hand_kd.data import ROOT, paired_sample
from hand_kd.losses import paired_loss
from hand_kd.utils import sha256_file, verify
from experiments.wilor_transfer.features import FeatureAligner, feature_loss


def load_wilor(root, device='cuda'):
    sys.path.insert(0, str(root / 'external/wilor'))
    from wilor.models import WiLoR
    from wilor.configs import get_config
    cfg = get_config(str(root / 'external/wilor/pretrained_models/model_config.yaml'))
    cfg.defrost()
    cfg.MODEL.BBOX_SHAPE = [192, 256]
    cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS', None)
    cfg.MANO.DATA_DIR = str(root / 'external/wilor/mano_data')
    cfg.MANO.MODEL_PATH = cfg.MANO.DATA_DIR
    cfg.MANO.MEAN_PARAMS = str(root / 'external/wilor/mano_data/mano_mean_params.npz')
    cfg.freeze()
    model = WiLoR(cfg, init_renderer=False)
    state = torch.load(root / 'models/wilor/wilor_final.ckpt', map_location='cpu', weights_only=False)
    model.load_state_dict(state['state_dict'], strict=True)
    del state
    return model.eval().requires_grad_(False).to(device), cfg


def training_mode(model, batchnorm_mode='train'):
    if batchnorm_mode not in ('train', 'released_stats'):
        raise ValueError('Unknown BatchNorm fine-tuning policy')
    model.train()
    if batchnorm_mode == 'released_stats':
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                if not module.track_running_stats: raise ValueError('Released statistics are unavailable')
                module.eval()


def checked_batch(path, digest):
    verify(path,digest)
    return torch.load(path,map_location='cpu',weights_only=True)


def prediction_parity(expected, actual):
    if expected.keys()!=actual.keys(): raise AssertionError('Output keys changed')
    for key in expected:
        torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=1e-6)
    return max(float((expected[k]-actual[k]).abs().max()) for k in expected)


def pilot_gate(candidate, control, initial):
    failures = []
    gain = control['mpjpe_mm']-candidate['mpjpe_mm']
    if not math.isfinite(gain) or gain < .1: failures.append('MPJPE gain below 0.1 mm')
    for label,reference in (('supervised',control),('initial',initial)):
        pairs = [(k,candidate[k],reference[k]) for k in ('mpjpe_mm','pa_mpjpe_mm','p95_joint_error_mm')]
        pairs += [(k,candidate['per_finger_mpjpe_mm'][k],v)
                  for k,v in reference['per_finger_mpjpe_mm'].items()]
        for key,value,ref in pairs:
            if not math.isfinite(value) or value > ref+.001: failures.append(f'{label}/{key} regression')
    return {'pass':not failures,'failures':failures,'gain_over_supervised_mm':gain,
            'scope':'exploratory fixed-update pilot, not full-epoch promotion or independent generalization'}


class DualPlan(Dataset):
    def __init__(self, base, indices, seed):
        self.base,self.indices,self.seed = base,np.asarray(indices),seed

    def __len__(self): return len(self.indices)

    def __getitem__(self,slot):
        py,np_state = random.getstate(),np.random.get_state()
        try:
            random.seed(self.seed+slot); np.random.seed(self.seed+slot)
            index = int(self.indices[slot])
            raw = self.base.raw(index)
            result = paired_sample(*raw,base_scale=1.5,teacher_view=True)
            result.update(original_sample_index=index,slot=slot,
                          source=0 if index<self.base.fh_count else 1,
                          source_rgb_sha256=hashlib.sha256(raw[0].tobytes()).hexdigest())
            return result
        finally:
            random.setstate(py); np.random.set_state(np_state)


def source_files():
    files = list((ROOT/'hand_kd').glob('*.py'))
    files += list((ROOT/'experiments/single_stack').glob('*.py'))
    files += [ROOT/'experiments/__init__.py']
    files += list((ROOT/'experiments/wilor_transfer').glob('*.py'))
    for repository in ('mobrecon','wilor'):
        files += list((ROOT/'external'/repository).rglob('*.py'))
    files += [ROOT/'external/wilor/pretrained_models/model_config.yaml',
              ROOT/'external/wilor/mano_data/mano_mean_params.npz']
    return sorted(set(files))


def verify_assets(config):
    for path,digest in ((config['initial_checkpoint'],config['initial_checkpoint_sha256']),
                        (config['teacher_checkpoint'],config['teacher_checkpoint_sha256']),
                        (config['dataset_manifest'],config['dataset_manifest_sha256']),
                        (config['dataset_selection'],config['dataset_selection_sha256'])):
        print(f'Checking {path}',flush=True); verify(ROOT/path,digest)
    manifest_path = Path(config['dataset_manifest'])
    manifest = json.loads(manifest_path.read_text())
    paths = [(manifest_path.parent/k,v) for k,v in manifest['files'].items()]
    paths += [(Path(v['path']),v['sha256']) for v in manifest['freihand']['arrays'].values()]
    paths += [(Path(manifest['freihand']['archive']),manifest['freihand']['sha256'])]
    paths += [(Path(v['path']),v['sha256']) for v in manifest['comphand']['archives'].values()]
    prior = config['validation']
    paths += [(ROOT/prior['manifest'],prior['manifest_sha256']),
              (ROOT/prior['cache']/'index.json',prior['cache_index_sha256'])]
    paths += [(ROOT/'external/mobrecon/template'/name,prior[key]) for name,key in
              (('j_reg.npy','regressor_sha256'),('right_faces.npy','topology_sha256'),
               ('transform.pkl','transform_sha256'))]
    for path,digest in paths:
        print(f'Checking {path}',flush=True); verify(path,digest)
    revisions = {}
    for name,key in (('mobrecon','upstream_revision'),('wilor','teacher_revision')):
        revision = subprocess.check_output(['git','-C',str(ROOT/'external'/name),'rev-parse','HEAD'],text=True).strip()
        if revision != config[key]: raise ValueError(f'{name} revision changed')
        revisions[name] = revision
    return {str(path):digest for path,digest in paths},revisions


def tensor_digest(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def make_head(arm, seed, prefit_checkpoint, device='cuda'):
    if arm not in ('supervised','fresh','prefitted'): raise ValueError('Unknown arm')
    if arm=='supervised': return None
    torch.manual_seed(seed)
    head=FeatureAligner(mode='region').to(device)
    if arm=='prefitted':
        head.load_state_dict(torch.load(prefit_checkpoint,map_location=device,weights_only=True),strict=True)
    return head.train().requires_grad_(True)


def comparison_gate(candidate, supervised, fresh, initial):
    controls={name:pilot_gate(candidate,value,initial) for name,value in
              [('supervised',supervised),('fresh',fresh)]}
    return {'pass':all(v['pass'] for v in controls.values()),'controls':controls}


def micro_slices(count, size):
    if count<=0 or size<=0:raise ValueError('Positive batch sizes required')
    for start in range(0,count,size):
        end=min(start+size,count)
        yield slice(start,end),(end-start)/count


def verify_stream_assets(assets):
    for path,digest in assets.items():
        if sha256_file(Path(path))!=digest:raise ValueError(f'Streamed training asset changed: {path}')


def backward_batch(model, head, taps, stored, teacher_features, faces, microbatch, feature_weight):
    """Accumulate the original per-sample mean objectives; caller owns optimizer.step."""
    totals={};n=len(stored['img'])
    for section,fraction in micro_slices(n,microbatch):
        batch={k:v[section].cuda() for k,v in stored.items() if isinstance(v,torch.Tensor)}
        predictions=[];features=[]
        for view in range(2):
            predictions.append(model(batch['img'][:,view*3:view*3+3]))
            if head:
                features.append(feature_loss(head(taps.features()),teacher_features[section,view]))
                taps.clear()
        paired=dict(verts=torch.cat([p['mesh_pred'] for p in predictions],-1),
                    joint_img=torch.cat([p['uv_pred'] for p in predictions],-1))
        losses=paired_loss(paired,batch,faces,epoch=9)
        kd=torch.stack(features).mean() if head else losses['loss'].new_zeros(())
        total=losses['loss']+feature_weight*kd
        if not torch.isfinite(total):raise FloatingPointError('Nonfinite objective')
        (total*fraction).backward()
        values={**losses,'feature':kd,'total':total}
        for key,value in values.items():totals[key]=totals.get(key,0.)+fraction*float(value.detach())
    return totals
