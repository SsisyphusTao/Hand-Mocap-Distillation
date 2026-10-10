"""Native paired objectives and normalized intermediate feature MSE."""

# Paired-view and zero-edge loss adapters follow HandMesh implementations.
# Source: https://github.com/SeanChenxy/HandMesh
# Revision: 3c87e958d4855f890e3884ec94bcfc0f99422c3d
#
# MIT License
#
# Copyright (c) 2021 chenxingyu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
from pathlib import Path
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'external/mobrecon'))


def feature_loss(predicted, teacher):
    if predicted.shape != teacher.shape: raise ValueError('Feature shape mismatch')
    def normalize(x):
        return F.layer_norm(x.float().permute(0,2,3,1),(x.shape[1],),eps=1e-5)
    return F.mse_loss(normalize(predicted),normalize(teacher.detach()))


def edge_length_loss(pred, gt, face, is_valid=None):
    """Upstream edge objective with a finite zero subgradient at coincident vertices.

    Positive squared lengths retain the original sqrt(sum(square)) values and derivatives.
    Evaluating sqrt at zero before masking would still backpropagate 0 * inf.
    Only zero squared lengths take this branch; NaN/Inf inputs remain visible to the guard.
    """
    def length(vertices, a, b):
        squared = ((vertices[:, face[:, a], :] - vertices[:, face[:, b], :]) ** 2).sum(2, keepdim=True)
        zero = squared == 0
        safe = torch.where(zero, torch.ones_like(squared), squared).sqrt()
        return torch.where(zero, torch.zeros_like(safe), safe)

    differences = [(length(pred, a, b) - length(gt, a, b)).abs()
                   for a, b in ((0, 1), (0, 2), (1, 2))]
    loss = torch.cat(differences, 1)
    if is_valid is not None:
        loss = loss * is_valid
    return loss.mean()


def paired_loss(prediction, batch, faces, epoch):
    """Same coefficients, reduction and warmup as MobRecon_DS.loss/runner.train."""
    from mobrecon.models.loss import l1_loss, normal_loss
    from mobrecon.models.loss import contrastive_loss_3d, contrastive_loss_2d
    vertices, uv = prediction['verts'], prediction['joint_img']
    truth = batch['verts']
    result = dict(verts_loss=l1_loss(vertices, truth),
                  joint_img_loss=l1_loss(uv, batch['joint_img']))
    result['normal_loss'] = .05 * (normal_loss(vertices[..., :3], truth[..., :3], faces)
                                   + normal_loss(vertices[..., 3:], truth[..., 3:], faces))
    result['edge_loss'] = .5 * (edge_length_loss(vertices[..., :3], truth[..., :3], faces)
                               + edge_length_loss(vertices[..., 3:], truth[..., 3:], faces))
    if epoch > 4:
        result['con3d_loss'] = contrastive_loss_3d(vertices, batch['aug_param'])
        result['con2d_loss'] = contrastive_loss_2d(uv, batch['bb2img_trans'], 128)
    # Match the author's summation order as well as the individual terms.
    result['loss'] = (result['verts_loss'] + result['normal_loss'] + result['edge_loss']
                      + result['joint_img_loss'] + result.get('con3d_loss', 0)
                      + result.get('con2d_loss', 0))
    return result


def paired_objective(model,batch,target,faces,epoch,weight):
    views=[model(batch['img'][:,i*3:i*3+3]) for i in range(2)]
    predicted=dict(verts=torch.cat([v['mesh_pred'] for v in views],-1),
                   joint_img=torch.cat([v['uv_pred'] for v in views],-1))
    losses=paired_loss(predicted,batch,faces,epoch)
    kd=sum(feature_loss(v['feature'],target[:,i]) for i,v in enumerate(views))/2 if weight else losses['loss'].new_zeros(())
    return {**losses,'feature':kd,'total':losses['loss']+weight*kd}


def teacher_targets(teacher,images):
    if teacher.training or any(p.requires_grad for p in teacher.parameters()):
        raise ValueError('Teacher must be frozen and in eval mode')
    features=[teacher.backbone(images[:,i*3:i*3+3])[0] for i in range(2)]
    value=torch.stack(features,1).detach()
    if value.shape[1:]!=(2,256,4,4) or not torch.isfinite(value).all():
        raise ValueError('Invalid teacher feature contract')
    return value
