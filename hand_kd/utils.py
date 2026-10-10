"""Hash, checkpoint, and deterministic-order helpers."""
import hashlib
import json
import os
from pathlib import Path
import random
import numpy as np
import torch


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


sha256_file = sha256


def tensor_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def epoch_indices(admitted, seed, epoch):
    return np.asarray(admitted)[np.random.default_rng(np.random.SeedSequence([seed,epoch])).permutation(len(admitted))]


def cpu_copy(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_copy(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_copy(v) for v in value)
    return value


def atomic_save(path, value):
    path=Path(path);temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        torch.save(value,stream);stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def load_bundle(path, config, sources):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['config']!=config or state['sources']!=sources:
        raise ValueError('Resume configuration/source identity mismatch')
    return state


def write_json(path, value):
    with path.open('x') as stream: json.dump(value, stream, indent=2)


def replace_json(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream: json.dump(value, stream, indent=2)
    temporary.replace(path)


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed); random.seed(seed)
    cv_threads = __import__('cv2')
    cv_threads.setNumThreads(0)


def verify(path, digest):
    if sha256_file(path) != digest: raise ValueError(f'Pinned input changed: {path}')


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
