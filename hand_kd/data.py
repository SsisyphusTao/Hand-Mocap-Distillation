"""Paired FreiHAND/CompHand training data and legacy labeled evaluation cache."""

# Paired-view preprocessing adapts HandMesh FreiHAND/CompHand implementations.
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

import io
import json
import os
from pathlib import Path
import random
import sys
import zipfile
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from hand_kd.utils import sha256, sha256_file

ROOT = Path(__file__).resolve().parents[1]
MANO_TO_MPII = [0, 13, 14, 15, 20, 1, 2, 3, 16, 4, 5, 6, 17, 10, 11, 12, 19, 7, 8, 9, 18]
TEACHER_CHECKPOINT_SHA256 = '3e97aafc7dd08d883a4cc5a027df61fdb6fda6136dbd1319405413862ada6bb2'
COMP_K = np.array([[373.3511425, 0, 128], [0, 373.3511425, 128], [0, 0, 1]])
sys.path.insert(0, str(ROOT/'external/mobrecon'))


def load_manifest(path):
    document = json.loads(Path(path).read_text())
    if document['schema_version'] != 1:
        raise ValueError('Unsupported manifest schema')
    seen, groups, image_splits = set(), {}, {}
    for sample in document['samples']:
        sid, split, group = sample['sample_id'], sample['split'], sample['group_id']
        if sid in seen or split not in ('train', 'validation', 'test'):
            raise ValueError('Duplicate sample or unknown split')
        seen.add(sid)
        if group in groups and groups[group] != split:
            raise ValueError('Group spans splits')
        groups[group] = split
        digest = sha256(sample['image_path'])
        if digest != sample['source_sha256']:
            raise ValueError(f'Source hash changed: {sid}')
        if digest in image_splits and image_splits[digest] != split:
            raise ValueError('Identical image spans splits')
        image_splits[digest] = split
        xyz, K = np.asarray(sample['joints_mpii_m']), np.asarray(sample['K'])
        if xyz.shape != (21, 3) or K.shape != (3, 3) or not np.isfinite(xyz).all() or not np.isfinite(K).all():
            raise ValueError(f'Invalid labeled geometry: {sid}')
        if sample['handedness'] != 'right' or (xyz[:, 2] <= 0).any():
            raise ValueError(f'Unsupported handedness or camera depth: {sid}')
    return document


def student_crop(image_bgr, box, crop_scale=1.5):
    box = np.asarray(box, np.float32)
    if box.shape != (4,) or not np.isfinite(box).all() or (box[2:] <= box[:2]).any() or crop_scale <= 0:
        raise ValueError('Invalid hand box/crop scale')
    center = (box[:2] + box[2:]) * .5
    side = float(max(box[2:] - box[:2]) * crop_scale)
    scale = 128 / side
    affine = np.array([[scale, 0, 64-scale*center[0]], [0, scale, 64-scale*center[1]]], np.float32)
    crop = cv2.warpAffine(image_bgr, affine, (128, 128), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    rgb = crop[..., ::-1].copy().astype(np.float32) / 255
    return ((rgb - .5) / .5).transpose(2, 0, 1), affine


def project_points(xyz, K):
    xyz, K = np.asarray(xyz), np.asarray(K)
    if xyz.shape[-1] != 3 or not np.isfinite(xyz).all() or (xyz[..., 2] <= 0).any():
        raise ValueError('Invalid camera-space points')
    projected = xyz @ K.T
    return projected[..., :2] / projected[..., 2:3]


def usable_uv_mask(uv_source, uv_crop, image_size, annotation_mask):
    return (np.asarray(annotation_mask,dtype=bool)
            & np.isfinite(uv_source).all(axis=1) & np.isfinite(uv_crop).all(axis=1)
            & np.all((uv_source >= 0) & (uv_source < np.asarray(image_size)),axis=1)
            & np.all((uv_crop >= 0) & (uv_crop < 1),axis=1))


def load_cache(manifest_path, cache_dir, regressor_path):
    manifest = load_manifest(manifest_path)
    cache_dir = Path(cache_dir)
    cache = json.loads((cache_dir / 'index.json').read_text())
    if cache['schema_version'] != 1 or cache['manifest_sha256'] != sha256(manifest_path):
        raise ValueError('Stale teacher manifest/schema')
    if cache['regressor_sha256'] != sha256(regressor_path):
        raise ValueError('Stale joint regressor')
    if cache['teacher_checkpoint_sha256'] != TEACHER_CHECKPOINT_SHA256:
        raise ValueError('Stale teacher checkpoint identity')
    records = cache['samples']
    if set(records) != {s['sample_id'] for s in manifest['samples']}:
        raise ValueError('Teacher cache sample set mismatch')
    for s in manifest['samples']:
        record = records[s['sample_id']]
        if record['source_sha256'] != s['source_sha256']:
            raise ValueError('Stale teacher source hash')
        if record['valid']:
            file = cache_dir / record['target_file']
            if sha256(file) != record['target_sha256']:
                raise ValueError('Teacher target hash changed')
            with np.load(file, allow_pickle=False) as values:
                if str(values['teacher_checkpoint_sha256']) != cache['teacher_checkpoint_sha256']:
                    raise ValueError('Stale teacher checkpoint')
                if values['vertices_m'].shape != (778, 3) or not np.isfinite(values['vertices_m']).all():
                    raise ValueError('Invalid teacher vertices')
    return manifest, cache


class PilotDataset:
    """CPU samples with a common frozen detection mask for both training arms."""
    def __init__(self, manifest_path, cache_dir, regressor_path, split):
        manifest, self.cache = load_cache(manifest_path, cache_dir, regressor_path)
        self.cache_dir = Path(cache_dir)
        self.samples = [s for s in manifest['samples'] if s['split'] == split and self.cache['samples'][s['sample_id']]['valid']]
        if not self.samples:
            raise ValueError(f'No valid {split} samples')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        record = self.cache['samples'][sample['sample_id']]
        image = cv2.imread(sample['image_path'])
        crop, affine = student_crop(image, record['box_xyxy'])
        xyz = np.asarray(sample['joints_mpii_m'], np.float32)
        uv_source = project_points(xyz, np.asarray(sample['K'], np.float32))
        uv = (np.c_[uv_source, np.ones(21)] @ affine.T / 128).astype(np.float32)
        visible = usable_uv_mask(uv_source,uv,[image.shape[1],image.shape[0]],sample.get('visibility',np.ones(21,dtype=bool)))
        with np.load(self.cache_dir / record['target_file'], allow_pickle=False) as target:
            vertices = target['vertices_m'].copy()
        return dict(image_student=crop, teacher_vertices_m=vertices,
                    joints_mpii_m=xyz-xyz[:1], uv=uv, uv_visible=visible,
                    sample_id=sample['sample_id'])


def project_points_upstream(points, K):
    points = np.asarray(points).reshape(-1, 3)
    projected = np.matmul(K, points.T).T
    return projected[:, :2] / projected[:, 2:]


def square_bbox(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError('Empty foreground mask')
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    side = max(w, h)
    return [x + .5*w - .5*side, y + .5*h - .5*side, side, side]


def paired_sample(image, mask, vertices, joints, K, color=True, base_scale=1.3, teacher_view=False):
    """Two independent original-image warps; 3D labels are meters before /0.2."""
    bbox = square_bbox(mask)
    uv = project_points_upstream(joints, K)
    principal = K[:2, 2].astype(np.float32)
    focal = np.array([K[0, 0], K[1, 1]], np.float32)
    from utils.preprocessing import augmentation, augmentation_2d
    from utils.augmentation import Augmentation
    from utils.vis import base_transform
    photo = Augmentation() if color else None
    views = []
    teacher_views = []
    for _ in range(2):
        roi, transform, inverse, params, flip, _, roi_mask = augmentation(
            image.copy(), bbox, 'train', exclude_flip=True, input_img_shape=(128, 128),
            mask=mask.copy(), base_scale=base_scale, scale_factor=.2, rot_factor=90,
            shift_wh=bbox[2:], gaussian_std=3)
        if teacher_view:
            teacher_roi = cv2.warpAffine(image.copy(), transform * 2, (256, 256),
                                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        if photo is not None:
            photo_state = np.random.get_state() if teacher_view else None
            roi = photo(roi)
            if teacher_view:
                after_photo = np.random.get_state()
                np.random.set_state(photo_state)
                teacher_roi = photo(teacher_roi)
                np.random.set_state(after_photo)
        if teacher_view:
            teacher_views.append(torch.from_numpy(
                (teacher_roi.astype(np.float32)/255).transpose(2,0,1).copy()))
        roi = torch.from_numpy(base_transform(roi, 128, mean=.5, std=.5)).float()
        params = torch.from_numpy(params).float()
        uv_view, principal_view = augmentation_2d(image, uv, principal, transform, flip)
        angle = np.deg2rad(-params[0].item())
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1]], np.float32)
        rotated_joints = torch.from_numpy(rotation @ joints.T).T.float()
        rotated_vertices = torch.from_numpy(rotation @ vertices.T).T.float()
        calib = np.eye(4)
        focal_view = focal * roi.size(1) / (bbox[2] * params[1])
        calib[0, 0], calib[1, 1] = focal_view
        calib[:2, 2] = principal_view
        views.append(dict(img=roi, mask=torch.from_numpy(roi_mask).float().unsqueeze(0),
                          calib=torch.from_numpy(calib).float(), verts=rotated_vertices,
                          joint_cam=rotated_joints, joint_img=torch.from_numpy(uv_view[:, :2]).float()/128,
                          aug_param=params, bb2img_trans=torch.from_numpy(inverse).float()))
    out = {name: torch.cat([v[name] for v in views], dim=(0 if name in
           ('img', 'mask', 'calib', 'aug_param') else -1)) for name in views[0]}
    out['root'] = out['joint_cam'][0].clone()
    out['joint_cam'] = (out['joint_cam'] - out['root']) / .2
    out['verts'] = (out['verts'] - out['root']) / .2
    if teacher_view:
        out['teacher_rgb'] = torch.cat(teacher_views, 0)
    return out


class ZipStore:
    """Each DataLoader worker owns its ZIP handles, including after fork."""
    def __init__(self):
        self.pid = None
        self.handles = {}

    def read(self, archive, member):
        if self.pid != os.getpid():
            for handle in self.handles.values(): handle.close()
            self.handles = {}; self.pid = os.getpid()
        if archive not in self.handles:
            self.handles[archive] = zipfile.ZipFile(archive)
        return self.handles[archive].read(member)


class CombinedDataset(Dataset):
    """All four FreiHAND backgrounds, followed by accepted CompHand images."""
    def __init__(self, manifest, color=True, crop_base_scale=1.3):
        self.path = Path(manifest)
        self.manifest = json.loads(self.path.read_text())
        self.records = json.loads((self.path.parent/'comphand-records.json').read_text())
        self.store = ZipStore()
        self.color = color
        self.crop_base_scale = float(crop_base_scale)
        if not np.isfinite(self.crop_base_scale) or self.crop_base_scale <= 0:
            raise ValueError('Crop base scale must be finite and positive')
        fh = self.manifest['freihand']
        self.K = np.load(fh['arrays']['K.npy']['path'], mmap_mode='r')
        self.joints = np.load(fh['arrays']['gt_joints.npy']['path'], mmap_mode='r')
        self.vertices = np.load(fh['arrays']['gt_vertices.npy']['path'], mmap_mode='r')
        self.comp_vertices = np.load(self.path.parent/'comphand-vertices.npy', mmap_mode='r')
        self.regressor = np.load(ROOT/'external/mobrecon/template/j_reg.npy')
        self.fh_count = fh['images']

    def __len__(self):
        return self.fh_count + len(self.records)

    def raw(self, index):
        if index < 0 or index >= len(self): raise IndexError(index)
        if index < self.fh_count:
            base = index % 32560
            archive = self.manifest['freihand']['archive']
            image = np.array(Image.open(io.BytesIO(self.store.read(archive, f'training/rgb/{index:08d}.jpg'))).convert('RGB'))
            mask = np.array(Image.open(io.BytesIO(self.store.read(archive, f'training/mask/{base:08d}.jpg'))).convert('RGB'))[..., 0]
            # The author's JSON annotation path computes in float64 before casting.
            return image, mask, self.vertices[base].copy(), self.joints[base].astype(np.float64), self.K[base].astype(np.float64)
        record = self.records[index-self.fh_count]
        archive = self.manifest['comphand']['archives'][record['archive']]['path']
        image = cv2.imdecode(np.frombuffer(self.store.read(archive, record['image']), np.uint8), cv2.IMREAD_COLOR)
        mask = cv2.imdecode(np.frombuffer(self.store.read(archive, record['mask']), np.uint8), cv2.IMREAD_COLOR)
        if image is None or mask is None: raise ValueError(f'Failed to decode {record}')
        # Mirrors from upstream, including the image's BGR -> RGB conversion.
        image, mask = image[:, ::-1, ::-1], mask[..., ::-1, 0]
        vertices = self.comp_vertices[record['mesh_index']].copy()
        from mobrecon.tools.kinematics import mano_to_mpii
        joints = mano_to_mpii(self.regressor @ vertices)
        return image, mask, vertices, joints, COMP_K.copy()

    def __getitem__(self, index):
        result = paired_sample(*self.raw(index), color=self.color, base_scale=self.crop_base_scale)
        result['sample_index'] = index
        result['source'] = 0 if index < self.fh_count else 1
        return result


class SelectionDataset(Dataset):
    def __init__(self, base, indices):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        if (self.indices.ndim != 1 or not len(self.indices) or
                self.indices[0] < 0 or self.indices[-1] >= len(base) or
                (np.diff(self.indices) <= 0).any()):
            raise ValueError('Selection must be sorted, unique, nonempty and in range')
        self.fh_count = int((self.indices < base.fh_count).sum())
        self.records = base.records

    def __len__(self): return len(self.indices)

    def __getitem__(self, index):
        result = dict(self.base[int(self.indices[index])])
        result['original_sample_index'] = result['sample_index']
        result['sample_index'] = index
        return result


def load_selection(base, path, expected_sha256):
    path = Path(path)
    if sha256_file(path) != expected_sha256:
        raise ValueError('Selection manifest changed')
    selection = json.loads(path.read_text())
    if sha256_file(base.path) != selection['parent_manifest_sha256']:
        raise ValueError('Selection parent dataset changed')
    index_path = path.parent/selection['indices_file']
    if sha256_file(index_path) != selection['indices_sha256']:
        raise ValueError('Selection indices changed')
    indices = np.load(index_path, allow_pickle=False)
    excluded = np.asarray(selection['excluded_original_indices'], dtype=np.int64)
    expected = np.setdiff1d(np.arange(len(base)), excluded)
    if (selection['offered'] != len(base) or len(set(excluded)) != len(excluded)
            or (excluded < 0).any() or (excluded >= len(base)).any()
            or not np.array_equal(indices, expected) or len(indices) != selection['admitted']):
        raise ValueError('Selection is not exactly the declared complement')
    return SelectionDataset(base, indices)


class PlannedDataset(Dataset):
    """Per-slot seeds keep augmentation identical across normalization/source arms."""
    def __init__(self, base, indices, seed):
        self.base, self.indices, self.seed = base, np.asarray(indices), int(seed)

    def __len__(self): return len(self.indices)

    def __getitem__(self, slot):
        py, np_state = random.getstate(), np.random.get_state()
        try:
            seed = (self.seed + int(slot)) % 2**32
            random.seed(seed); np.random.seed(seed)
            return self.base[int(self.indices[slot])]
        finally:
            random.setstate(py); np.random.set_state(np_state)
