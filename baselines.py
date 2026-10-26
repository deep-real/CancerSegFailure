#!/usr/bin/env python3
"""Run the five paper baseline methods using the same segmentation checkpoint."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from medfoe.baselines import METHODS, predict_mask
from medfoe.data import read_manifest, read_image, evaluate_saved_masks
from medfoe.model import PICAISegmenter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/picai.json")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--threshold", type=float, help="Override the configured cutoff")
    parser.add_argument("--cleanup", action="store_true", help="Apply fixed 256-pixel removal and radius-2 erosion")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    checkpoint = (args.config.resolve().parent / config["segmentation_checkpoint"]).resolve()
    threshold = config["baseline_thresholds"][args.method] if args.threshold is None else args.threshold
    if not np.isfinite(threshold) or (args.method == "direct_threshold" and not 0 <= threshold <= 1):
        parser.error("Provide a finite score cutoff, or a pixel cutoff in [0,1]")
    cleanup = dict(minimum_component_pixels=256, boundary_offset_pixels=-2) if args.cleanup else dict(
        minimum_component_pixels=0, boundary_offset_pixels=0,
    )
    manifest = read_manifest(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "masks").mkdir()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = PICAISegmenter.from_checkpoint(checkpoint, args.device)
    with torch.inference_mode():
        for row, sample in enumerate(manifest["samples"]):
            logits = model(read_image(sample["path"]).to(args.device)[None])[0, 3]
            mask = predict_mask(logits.cpu().numpy(), args.method, threshold, cleanup,
                                logits.sigmoid().cpu().numpy(), temperature=1.0)
            np.savez_compressed(args.output_dir / "masks" / f"{row:05d}.npz", mask=mask)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output_dir / "config.json").write_text(json.dumps(dict(
        segmentation_checkpoint=str(checkpoint), method=args.method, threshold=threshold,
        cleanup=cleanup, temperature=1.0,
    ), indent=2) + "\n")
    if args.evaluate:
        print(json.dumps(evaluate_saved_masks(manifest["samples"], args.output_dir), indent=2))


if __name__ == "__main__":
    main()
