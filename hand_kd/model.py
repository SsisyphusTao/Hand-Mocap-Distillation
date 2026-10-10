"""Pinned HandMesh model and the single-stack student."""
import pickle
import sys
from copy import deepcopy
from types import SimpleNamespace
import torch
from torch import nn


def setup_torch(seed=0):
    import numpy as np
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def build_mobrecon(root):
    """Construct the pinned full architecture without requiring model weights."""
    upstream = root / 'external/mobrecon'
    sys.path.insert(0, str(upstream))
    from cmr.models.mobrecon_densestack import MobRecon
    from utils.utils import preprocess_spiral
    with (upstream / 'template/transform.pkl').open('rb') as stream:
        topology = pickle.load(stream, encoding='latin1')
    spirals = [preprocess_spiral(f, 9, v, 1) for f, v in zip(topology['face'][:-1], topology['vertices'][:-1])]
    up = []
    for transform in topology['up_transform']:
        coo = transform.tocoo()
        up.append((torch.tensor(coo.row, dtype=torch.long), torch.tensor(coo.col, dtype=torch.long), torch.tensor(coo.data, dtype=torch.float32)))
    return MobRecon(SimpleNamespace(out_channels=[32, 64, 128, 256], dsconv=False), spirals, up)


def load_mobrecon(root, checkpoint, device='cuda'):
    model = build_mobrecon(root)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    return model.to(device)


class SingleStackMobRecon(nn.Module):
    def __init__(self, full_model):
        super().__init__()
        self.backbone = deepcopy(full_model.backbone)
        self.decoder3d = deepcopy(full_model.decoder3d)
        for name in ('dense_stack1', 'stack1_remap', 'thrink2'):
            delattr(self.backbone, name)

    def forward(self, image):
        b = self.backbone
        # Keep the same image lattice and channel width at the retained stack.
        inputs = b.thrink(b.reorg(b.pre_layer(image)))
        output, middle = b.dense_stack2(inputs)
        feature = b.mid_proj(middle)
        uv = b.uv_reg(b.conv(output).reshape(len(image), 21, -1))
        return dict(mesh_pred=self.decoder3d(uv, feature), uv_pred=uv, feature=feature)


def load_single_stack(root, checkpoint, device='cuda'):
    """Strict student-only loading; no teacher checkpoint or feature head needed."""
    model = SingleStackMobRecon(build_mobrecon(root))
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    return model.to(device)


def verify_reload(model, reloaded, sample):
    """Require identical state; allow native CUDA scatter floating-point noise."""
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, reloaded.state_dict()[name], rtol=0, atol=0)
    model.eval()
    reloaded.eval()
    before, after = model(sample), reloaded(sample)
    errors = {key: float((before[key]-after[key]).abs().max()) for key in before}
    for key in before:
        # 1e-5 native mesh units is 0.002 mm after the fixed x0.20 conversion.
        tolerance = 1e-5 if key == 'mesh_pred' else 0.
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=tolerance)
    return errors
