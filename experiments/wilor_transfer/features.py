"""Training-only feature alignment inspired by PAT (Lin et al., ICCV 2025).

Independent implementation of cross-stage regional self-attention, equations 3-4.
This bounded adaptation uses one pooled teacher level, normalized MSE, no AFP,
no output KL and no HCL. It is not a reproduction of the complete PAT method.
"""
import torch
from torch import nn
from torch.nn import functional as F

STUDENT_LAYERS = ('backbone.pre_layer', 'backbone.thrink',
                  'backbone.dense_stack1', 'backbone.mid_proj')
STUDENT_CHANNELS = (128, 128, 24, 256)


def teacher_input(rgb, mean, std):
    """RGB floats, same photometric augmentation; reproduce WiLoR's central trim."""
    if rgb.ndim != 4 or rgb.shape[1:] != (3,256,256):
        raise ValueError('Expected Bx3x256x256 RGB')
    mean = rgb.new_tensor(mean).reshape(1,3,1,1)
    std = rgb.new_tensor(std).reshape(1,3,1,1)
    if (std <= 0).any(): raise ValueError('Positive normalization std required')
    return ((rgb-mean)/std)[:,:,:,32:-32]


def student_regions(feature, grid=(8,6)):
    """Associate normalized feature regions with teacher's central 75% crop.

    Uses feature-map extent coordinates, not a claim of identical receptive
    fields/token centers across CNN and padded ViT convolutions.
    """
    h,w = grid
    y = (torch.arange(h,device=feature.device,dtype=feature.dtype)+.5)*2/h-1
    x = ((torch.arange(w,device=feature.device,dtype=feature.dtype)+.5)*2/w-1)*.75
    yy,xx = torch.meshgrid(y,x,indexing='ij')
    locations = torch.stack((xx,yy),-1).unsqueeze(0).expand(len(feature),-1,-1,-1)
    return F.grid_sample(feature,locations,mode='bilinear',padding_mode='border',align_corners=False)


class StudentTaps:
    """Read-only forward hooks; all handles and graphs can be explicitly released."""
    def __init__(self, model, names=STUDENT_LAYERS):
        self.names = tuple(names)
        self.values = {}
        self.handles = []
        modules = dict(model.named_modules())
        for name in self.names:
            self.handles.append(modules[name].register_forward_hook(
                lambda module, args, output, key=name: self.values.__setitem__(key,output)))

    def features(self):
        return [self.values[name] for name in self.names]

    def clear(self): self.values.clear()

    def close(self):
        for handle in self.handles: handle.remove()
        self.handles.clear(); self.clear()


class FeatureAligner(nn.Module):
    def __init__(self, channels=STUDENT_CHANNELS, teacher_channels=1280,
                 width=128, grid=(8,6), mode='projected'):
        super().__init__()
        if mode not in ('projected','region'): raise ValueError('Unknown alignment mode')
        self.grid,self.mode = tuple(grid),mode
        self.projections = nn.ModuleList([nn.Conv2d(c,width,1) for c in channels])
        self.norm = nn.LayerNorm(width)
        self.position = nn.Parameter(torch.zeros(1,len(channels)*grid[0]*grid[1],width))
        self.output = nn.Conv2d(width,teacher_channels,1)
        # Common projection/readout parameters are initialized before attention.
        self.attention = nn.MultiheadAttention(width,4,dropout=0,batch_first=True) if mode=='region' else None

    def forward(self, features):
        if len(features) != len(self.projections): raise ValueError('Wrong number of stages')
        tokens = [projection(student_regions(feature,self.grid)).flatten(2).transpose(1,2)
                  for feature,projection in zip(features,self.projections)]
        x = self.norm(torch.cat(tokens,1)+self.position)
        if self.attention is not None:
            mixed,_ = self.attention(x,x,x,need_weights=False)
            x = x+mixed
        b,_,c = x.shape
        # Both arms have the same stage aggregation and output projection.
        x = x.reshape(b,len(features),*self.grid,c).mean(1).permute(0,3,1,2)
        return self.output(x)


def feature_loss(predicted, teacher):
    if predicted.shape != teacher.shape: raise ValueError('Feature shape mismatch')
    def normalize(x):
        return F.layer_norm(x.float().permute(0,2,3,1),(x.shape[1],),eps=1e-5)
    return F.mse_loss(normalize(predicted),normalize(teacher.detach()))
