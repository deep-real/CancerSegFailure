"""Prepared MRI slices, complete-examination manifests and saved-mask scoring."""

import json
import re
from pathlib import Path

import numpy as np


def patient_id(identifier):
    """Accept a patient ID or a PI-CAI patient_study identifier.

    Only the documented PI-CAI numeric format is parsed. Arbitrary identifiers
    containing underscores must not be truncated.
    """
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError('A nonempty patient identifier is required')
    match = re.fullmatch(r'(\d{5})_\d{7}', identifier)
    return match.group(1) if match else identifier


def patient_key(row):
    return patient_id(row['patient_id'])


def study_key(row):
    """Neighbor context belongs to one examination, even for repeat patients."""
    return row.get('study_id', row['patient_id'])


def read_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    rows = manifest['samples']
    if not rows:
        raise ValueError('Manifest contains no samples')
    seen = set()
    for row in rows:
        if not all(isinstance(row.get(key), str) and row[key] for key in ('path', 'patient_id', 'slice_id')):
            raise ValueError('Each sample needs path, patient_id and slice_id strings')
        key = (study_key(row), row['slice_id'])
        if key in seen:
            raise ValueError('Manifest contains duplicate examination/slice pairs')
        seen.add(key)
        source = Path(row['path'])
        row['path'] = str(source if source.is_absolute() else path.parent/source)
        if not Path(row['path']).is_file():
            raise FileNotFoundError(row['path'])
    return manifest


def read_image(path):
    """Resize and normalize MRI without reading segmentation labels."""
    import h5py
    import torch
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.functional import resize
    with h5py.File(path, 'r') as source:
        image = torch.as_tensor(np.asarray(source['ct']), dtype=torch.float32)
    if image.ndim != 3 or image.shape[0] != 3 or not torch.isfinite(image).all():
        raise ValueError('Expected finite ct [3,H,W]')
    image = resize(image, [1024, 1024], interpolation=InterpolationMode.BILINEAR, antialias=True)
    maximum = image.amax(dim=(-2, -1), keepdim=True)
    return (image/torch.where(maximum != 0, maximum, torch.ones_like(maximum))).clamp_min_(0)


def read_target(path):
    """Resize binary cancer labels with nearest-neighbor interpolation."""
    import h5py
    import torch
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.functional import resize
    with h5py.File(path, 'r') as source:
        mask = np.asarray(source['seg'])
    if mask.ndim != 2 or not np.isfinite(mask).all() or not np.isin(mask, [0, 1]).all():
        raise ValueError('Expected finite binary lesion labels')
    return resize(torch.as_tensor(mask)[None].float(), [1024, 1024],
                  interpolation=InterpolationMode.NEAREST)[0].bool().numpy()


def examinations(samples):
    """Return original manifest indices in complete examination/slice order."""
    cases = []
    for study in sorted({study_key(row) for row in samples}):
        ids = sorted(
            (i for i, row in enumerate(samples) if study_key(row) == study),
            key=lambda i: int(samples[i]["slice_id"]),
        )
        if [int(samples[i]["slice_id"]) for i in ids] != list(range(len(ids))):
            raise ValueError(
                "Each examination must contain all consecutive slices, numbered from 0"
            )
        if len({patient_key(samples[i]) for i in ids}) != 1:
            raise ValueError("An examination cannot contain different patients")
        cases.append(ids)
    return cases


def metrics(statistics):
    """Slice-presence F1 and mean Dice over all target-positive slices."""
    area, truth, hit = np.asarray(statistics).T
    positive, detected = truth > 0, area > 0
    tp, fp, fn, tn = [int(value.sum()) for value in (
        positive & detected, ~positive & detected,
        positive & ~detected, ~positive & ~detected,
    )]
    dice = np.divide(2 * hit, area + truth, out=np.zeros(len(area), float), where=(area + truth) > 0)
    return dict(
        f1=2 * tp / max(2 * tp + fp + fn, 1),
        positive_slice_dice=float(dice[positive].mean()) if positive.any() else None,
        tp=tp, fp=fp, fn=fn, tn=tn, slices=len(area), positive_slices=int(positive.sum()),
    )


def evaluate_saved_masks(samples, output):
    """Read references only after prediction has saved every output mask."""
    statistics = []
    for row, sample in enumerate(samples):
        with np.load(Path(output) / "masks" / f"{row:05d}.npz") as saved:
            mask = saved["mask"]
        target = read_target(sample["path"])
        statistics.append([mask.sum(), target.sum(), (mask & target).sum()])
    result = metrics(statistics)
    (Path(output) / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
