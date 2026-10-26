"""Trained SAE maps and single-concept mask reconstruction."""

import numpy as np
import torch

from medfoe.sae import normalize_embeddings


def revise_mask(codes, prediction, concept_id, top_k=20):
    """Reconstruct accepted foreground patches with a resized signed distance."""
    import cv2
    from scipy.ndimage import distance_transform_edt

    codes = np.asarray(codes, dtype=np.float32)
    prediction = np.asarray(prediction, dtype=bool)
    if (codes.ndim != 3 or min(codes.shape) < 1 or not np.isfinite(codes).all()
            or (codes < 0).any() or prediction.ndim != 2):
        raise ValueError("Expected finite nonnegative concept codes and a 2D prediction")
    rows, columns, count = codes.shape
    height, width = prediction.shape
    if height < rows or width < columns or height % rows or width % columns:
        raise ValueError("Prediction must align with the encoder patch grid")
    if not isinstance(concept_id, (int, np.integer)) or not 0 <= concept_id < count:
        raise ValueError(f"Concept ID must be an integer in [0, {count - 1}]")
    if not isinstance(top_k, (int, np.integer)) or not 1 <= top_k <= count:
        raise ValueError(f"Revision top-k must be in [1, {count}]")
    support = prediction.reshape(rows, height // rows, columns, width // columns).any((1, 3))
    values = codes[support]
    selected = values[:, concept_id]
    # Do not let a zero entry in a top-k result count as concept evidence.
    # Resolve ties by ascending ID so the rule is independent of torch/device.
    rank = 1 + ((values > selected[:, None]) | (
        (values == selected[:, None]) & (np.arange(count) < concept_id))).sum(axis=1)
    accepted = np.zeros((rows, columns), dtype=bool)
    accepted[support] = (selected > 0) & (rank <= top_k)
    if not accepted.any():
        revised = np.zeros_like(prediction)
    elif accepted.all():
        revised = np.ones_like(prediction)
    else:
        distance = distance_transform_edt(accepted) - distance_transform_edt(~accepted)
        enlarged = cv2.resize(distance.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
        revised = enlarged > 0
    return dict(
        concept_id=int(concept_id), top_k=int(top_k), positive_activation_required=True,
        tie_policy="ascending concept ID", reconstruction="bilinear signed distance, threshold > 0",
        predicted_patches=int(support.sum()), accepted_patches=int(accepted.sum()),
        raw_pixels=int(prediction.sum()), revised_pixels=int(revised.sum()),
        removed_pixels=int((prediction & ~revised).sum()), added_pixels=int((revised & ~prediction).sum()),
        grid=accepted, mask=revised, activation=codes[:, :, concept_id].copy(),
    )


def relative_activation(maps):
    """Normalize display copies; zero maps stay transparent."""
    maps = np.asarray(maps, dtype=np.float32)
    if maps.ndim != 3 or not np.isfinite(maps).all() or (maps < 0).any():
        raise ValueError("Expected finite nonnegative maps [concept, row, column]")
    maxima = maps.max(axis=(1, 2), keepdims=True)
    return np.divide(maps, maxima, out=np.zeros_like(maps), where=maxima > 0)


def notebook_activation(maps, shape):
    """Apply cubic interpolation, Gaussian smoothing and min-max scaling.

    This is a display copy. Neither the saved codes nor mask decisions use it.
    """
    from scipy.ndimage import gaussian_filter, zoom

    maps = np.asarray(maps, dtype=np.float32)
    relative_activation(maps)  # Validate before interpolating.
    output = np.zeros((len(maps), *shape), dtype=np.float32)
    for i, raw in enumerate(maps):
        # Constant maps have no spatial contrast; do not amplify rounding noise.
        if raw.max() == raw.min():
            continue
        enlarged = zoom(raw, (shape[0] / raw.shape[0], shape[1] / raw.shape[1]), order=3)
        blurred = gaussian_filter(enlarged, 0.01 * max(shape))
        blurred -= blurred.min()
        if blurred.max() > 0:
            output[i] = blurred / blurred.max()
    return output


def mask_outline(mask):
    """One-pixel inner outline, including foreground on the image edge."""
    mask = np.asarray(mask, dtype=bool)
    padded = np.pad(mask, 1)
    interior = (padded[1:-1, :-2] & padded[1:-1, 2:]
                & padded[:-2, 1:-1] & padded[2:, 1:-1])
    return mask & ~interior


@torch.inference_mode()
def localize(config, input_path, concept_ids=(469, 968, 204, 1097), layer=11,
             step=6200, device="cpu", revise_with=None):
    """Read trained concepts from one MRI; reference masks are not used."""
    import json
    from pathlib import Path
    from medfoe import FEATURE_LAYERS
    from medfoe.data import read_image
    from medfoe.model import PICAISegmenter
    from medfoe.sae import SAE

    if layer not in FEATURE_LAYERS:
        raise ValueError(f"Choose layer from {FEATURE_LAYERS}")
    ids = np.asarray(concept_ids, dtype=np.int64)
    if ids.ndim != 1 or not 1 <= len(ids) <= 8 or ids.min() < 0 or ids.max() >= 1536 or len(np.unique(ids)) != len(ids):
        raise ValueError("Choose 1 to 8 distinct concept IDs between 0 and 1535")
    config = Path(config).resolve()
    settings = json.loads(config.read_text())
    checkpoint = config.parent / settings["segmentation_checkpoint"]
    weights = config.parent / settings["saes"][str(step)][str(layer)]
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = PICAISegmenter.from_checkpoint(checkpoint, device)
    sae = SAE.from_checkpoint(weights, device)
    image = read_image(input_path).to(device)
    probability = model(image[None])[0, 3].sigmoid().cpu().numpy()
    raw = probability > 0.5
    codes = sae.encode(normalize_embeddings(model.activations[layer][0])).float().cpu().numpy()
    result = dict(
        image=image.cpu().numpy(), probability=probability, raw_prediction=raw,
        concept_ids=ids, maps=codes[:, :, ids].transpose(2, 0, 1).copy(),
        layer=layer, step=step, threshold=float(sae.threshold),
    )
    if revise_with is not None:
        result["revision"] = revise_mask(codes, raw, int(revise_with), top_k=20)
    return result


def draw_outline(ax, mask, color):
    pixels = np.zeros((*mask.shape, 4), dtype=np.float32)
    pixels[mask_outline(mask)] = (*color, 1)
    ax.imshow(pixels, interpolation="nearest")


def plot_concepts(result, channel=0):
    """Plot smoothed concept maps with a jet overlay."""
    import matplotlib.pyplot as plt

    image = result["image"][channel]
    maps = notebook_activation(result["maps"], image.shape)
    figure, axes = plt.subplots(1, len(maps) + 1, figsize=(3 * (len(maps) + 1), 3.4), squeeze=False)
    axes = axes[0]
    for ax in axes:
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    draw_outline(axes[0], result["raw_prediction"], (0, 0.85, 1))
    axes[0].set_title("MRI and raw mask")
    for ax, activation, concept in zip(axes[1:], maps, result["concept_ids"]):
        pixels = plt.get_cmap("jet")(activation)
        pixels[..., 3] = activation ** 0.7
        ax.imshow(pixels)
        ax.set_title(f"Concept {concept}")
    figure.suptitle(f"MedFoE · layer {result['layer']} · SAE step {result['step']}")
    figure.tight_layout()
    return figure


def plot_revision(result, channel=0):
    """Show a single-concept reconstruction and changed pixels."""
    import matplotlib.pyplot as plt

    image = result["image"][channel]
    raw = result["raw_prediction"]
    revised = result["revision"]["mask"]
    figure, axes = plt.subplots(1, 3, figsize=(9, 3.4))
    for ax in axes:
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    draw_outline(axes[0], raw, (0, 0.85, 1))
    draw_outline(axes[1], revised, (1, 0, 0.7))
    changes = np.zeros((*raw.shape, 4), dtype=np.float32)
    changes[raw & ~revised] = (1, 0.2, 0.2, 0.7)
    changes[revised & ~raw] = (1, 0.75, 0, 0.7)
    axes[2].imshow(changes)
    for ax, title in zip(axes, ("Raw mask", "Concept reconstruction", "Removed (red) / added (gold)")):
        ax.set_title(title)
    figure.suptitle(f"Concept {result['revision']['concept_id']} · top-20 positive patch support")
    figure.tight_layout()
    return figure
