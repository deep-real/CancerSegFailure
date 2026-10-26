"""The five output-based baseline methods reported in the paper."""

import numpy as np
from scipy.special import expit, xlogy

from medfoe.postprocess import crop_bounds, refine_mask

METHODS = ("direct_threshold", "max_prob", "mean_prob", "entropy", "energy")


def slice_scores(logits, probability=None, temperature=1.0):
    """Whole-foreground scores; lower means more confident for every method.

    Energy uses binary logits [0,z], not the four overlapping anatomical heads.
    Its sign and scale follow -T*logsumexp(logits/T). Empty predictions have
    zero placeholders; their masks must remain empty regardless of the cutoff.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or not np.isfinite(logits).all():
        raise ValueError('A finite 2D tumor-logit image is required')
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Positive finite temperature required')
    probability = expit(logits) if probability is None else np.asarray(probability)
    if (probability.shape != logits.shape or not np.isfinite(probability).all()
            or (probability < 0).any() or (probability > 1).any()):
        raise ValueError('Aligned probabilities in [0,1] required')
    foreground = probability > .5
    if not foreground.any():
        return np.zeros(4, np.float64)
    p = probability[foreground].astype(np.float64)
    entropy = -(xlogy(p, p) + xlogy(1-p, 1-p)) / np.log(2)
    energy = -temperature * np.logaddexp(0, logits[foreground]/temperature)
    return np.array([1-p.max(), 1-p.mean(), entropy.mean(), energy.mean()])


def process_mask(mask, settings):
    """Apply the same label-free mask cleanup to any baseline."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError('A 2D mask is required')
    result = np.zeros_like(mask)
    box = crop_bounds(mask, abs(settings['boundary_offset_pixels'])+1)
    if box is not None:
        result[box] = refine_mask(mask[box], **settings)
    return result


def predict_mask(logits, method, threshold, settings, probability=None, temperature=1.0):
    """Return an actual prediction mask without accepting reference labels."""
    if method not in METHODS or not np.isfinite(threshold):
        raise ValueError('Known method and finite threshold required')
    probability = expit(logits) if probability is None else np.asarray(probability)
    scores = slice_scores(logits, probability, temperature)
    if method == 'direct_threshold':
        if not 0 <= threshold <= 1:
            raise ValueError('Pixel threshold must be in [0,1]')
        mask = probability > threshold
    else:
        score = scores[METHODS.index(method)-1]
        mask = (probability > .5) & (score <= threshold)
    return process_mask(mask, settings)
