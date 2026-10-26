#!/usr/bin/env python3
"""Run frozen MedFoE prediction and optional saved-mask evaluation."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from medfoe import FEATURE_LAYERS
from medfoe.model import PICAISegmenter
from medfoe.sae import SAE, normalize_embeddings
from medfoe.data import read_manifest, read_image, examinations, evaluate_saved_masks
from medfoe.postprocess import (boundary_features, choose_radii, predict_gains, preserve_empty_regions, corrected_mask, retained_samples, rescue_empty_slices, component_support, crop_bounds, mask_from_patches, refine_mask, geometry_features, partition, calibrated_keep)


def load_models(settings, device):
    segmenter = PICAISegmenter.from_checkpoint(settings["segmentation_checkpoint"], device)
    saes = {
        step: {int(layer): SAE.from_checkpoint(path, device) for layer, path in layers.items()}
        for step, layers in settings["saes"].items()
    }
    detectors = {
        "6200": {name: joblib.load(path) for name, path in settings["primary_detectors"].items()},
        "10000": joblib.load(settings["proposal_detectors"]),
    }
    return segmenter, saes, detectors, joblib.load(settings["boundary_model"])


def read_settings(path):
    path = Path(path).resolve()
    settings = json.loads(path.read_text())

    def resolve(value):
        return str((path.parent / value).resolve())

    for key in (
        "segmentation_checkpoint",
        "proposal_detectors",
        "boundary_model",
        "method_config",
    ):
        settings[key] = resolve(settings[key])
    settings["primary_detectors"] = {
        key: resolve(value) for key, value in settings["primary_detectors"].items()
    }
    settings["saes"] = {
        step: {layer: resolve(value) for layer, value in layers.items()}
        for step, layers in settings["saes"].items()
    }
    if set(settings["saes"]) != {"6200", "10000"} or any(
        set(layers) != set(map(str, FEATURE_LAYERS))
        for layers in settings["saes"].values()
    ):
        raise ValueError(
            "Provide the six SAE checkpoints for both steps 6200 and 10000"
        )
    return settings


def detect_slice(segmenter, saes, detectors, image):
    """Read image activations once; no annotation enters either SAE readout."""
    prediction = (segmenter(image[None])[0, 3].sigmoid() > 0.5).cpu().numpy()
    coords = np.argwhere(prediction.reshape(64, 16, 64, 16).any(axis=(1, 3)))
    y, x = coords.T
    records = {}
    for step, layers in saes.items():
        record = dict(patch=np.empty(0), image=1.0, active_concepts=0, active_l11=0)
        if len(coords):
            encoded = (
                torch.cat(
                    [
                        layers[layer].encode(
                            normalize_embeddings(segmenter.activations[layer][0])[y, x]
                        )
                        for layer in FEATURE_LAYERS
                    ],
                    dim=1,
                )
                .cpu()
                .numpy()
            )
            pooled = encoded.max(axis=0)
            bundle = detectors[step]
            record = dict(
                patch=bundle["patch_detector"].predict_proba(
                    bundle["patch_scaler"].transform(encoded)
                )[:, 1],
                image=float(
                    bundle["image_detector"].predict_proba(
                        bundle["image_scaler"].transform(pooled[None])
                    )[0, 1]
                ),
                active_concepts=int((pooled > 0).sum()),
                active_l11=int((pooled[:1536] > 0).sum()),
            )
        records[step] = record
    return prediction, records


def correct_examination(predictions, records, method, boundary_model):
    """Aggregate spatial evidence and refine mask boundaries."""
    n = len(predictions)
    coords = [
        np.argwhere(mask.reshape(64, 16, 64, 16).any(axis=(1, 3)))
        for mask in predictions
    ]
    counts = np.array(list(map(len, coords)))
    if not counts.sum():
        return [mask.copy() for mask in predictions]
    occupied = np.zeros((n, 64, 64), bool)
    for row, xy in enumerate(coords):
        occupied[row, xy[:, 0], xy[:, 1]] = True
    inputs = dict(
        area=np.array([mask.sum() for mask in predictions]),
        coords=np.concatenate(coords),
        patch_slice=np.repeat(np.arange(n), counts),
        patch_area=np.concatenate(
            [
                mask.reshape(64, 16, 64, 16).sum((1, 3))[xy[:, 0], xy[:, 1]]
                for mask, xy in zip(predictions, coords)
            ]
        ),
        occupied=occupied,
        slice_ids=np.arange(n),
        studies=[np.arange(n)],
    )
    contexts = {}
    for step in ("6200", "10000"):
        contexts[step] = dict(
            inputs,
            patch=np.concatenate([r[step]["patch"] for r in records]),
            image=np.array([r[step]["image"] for r in records]),
            active_concepts=np.array([r[step]["active_concepts"] for r in records]),
            active_l11=np.array([r[step]["active_l11"] for r in records]),
        )
    topology = component_support(contexts["6200"])
    groups = partition(inputs, topology["component_of_patch"])
    area = np.zeros((len(groups["sample_slice"]), 3), np.int64)
    clean_patch_area = np.zeros(len(inputs["patch_slice"]), np.int64)
    for row, mask in enumerate(predictions):
        if not mask.any():
            continue
        ids = np.flatnonzero(inputs["patch_slice"] == row)
        y, x = inputs["coords"][ids].T
        box = crop_bounds(mask, 3)
        for radius in range(3):
            clean = np.zeros_like(mask)
            clean[box] = refine_mask(
                mask[box], minimum_component_pixels=256, boundary_offset_pixels=-radius
            )
            values = clean.reshape(64, 16, 64, 16).sum((1, 3))[y, x]
            np.add.at(area[:, radius], groups["patch_sample"][ids], values)
            if radius == 2:
                clean_patch_area[ids] = values
    primary = calibrated_keep(contexts["6200"], method["reference"], topology)
    proposal = calibrated_keep(contexts["10000"], method["proposal"], topology)
    keep, _ = rescue_empty_slices(
        primary, proposal, clean_patch_area, inputs["patch_slice"], n
    )
    raw_area = np.bincount(
        groups["patch_sample"], weights=inputs["patch_area"], minlength=len(area)
    )
    features = np.column_stack(
        (geometry_features(inputs, groups), boundary_features(area, raw_area))
    )
    gain = predict_gains(boundary_model, features, groups["sample_slice"], np.arange(n))
    radii = preserve_empty_regions(
        choose_radii(gain, 0), retained_samples(keep, groups), area
    )
    output = []
    for row, mask in enumerate(predictions):
        selected = mask_from_patches(mask, inputs, row, keep)
        base = np.zeros_like(mask)
        box = crop_bounds(selected, 3)
        if box is not None:
            base[box] = refine_mask(
                selected[box], minimum_component_pixels=256, boundary_offset_pixels=-2
            )
        output.append(corrected_mask(mask, base, inputs, row, keep, groups, radii))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/picai.json")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args(argv)
    settings = read_settings(args.config)
    method = json.loads(Path(settings["method_config"]).read_text())
    if (method["method"] != "medfoe"
            or method["fragment_removal_pixels"] != 256
            or method["reference_erosion_radius"] != 2):
        parser.error("Use the supplied configs/method.json")
    manifest = read_manifest(args.manifest)
    samples = manifest["samples"]
    cases = examinations(samples)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "masks").mkdir()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    segmenter, saes, detectors, boundary_model = load_models(settings, args.device)
    with torch.inference_mode():
        for number, ids in enumerate(cases, 1):
            predictions, records = [], []
            for row in ids:
                prediction, evidence = detect_slice(
                    segmenter, saes, detectors,
                    read_image(samples[row]["path"]).to(args.device),
                )
                predictions.append(prediction)
                records.append(evidence)
            masks = correct_examination(predictions, records, method, boundary_model)
            for row, mask in zip(ids, masks):
                np.savez_compressed(args.output_dir / "masks" / f"{row:05d}.npz", mask=mask)
            print(f"Predicted examination {number}/{len(cases)}", flush=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output_dir / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    if args.evaluate:
        print(json.dumps(evaluate_saved_masks(samples, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
