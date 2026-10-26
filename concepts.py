#!/usr/bin/env python3
"""Save SAE concept heatmaps and the optional notebook mask reconstruction."""

import argparse
import json
from pathlib import Path

import numpy as np

from medfoe.concepts import localize, plot_concepts, plot_revision


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/picai.json")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--concepts", nargs="+", type=int, default=[469, 968, 204, 1097])
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--step", type=int, default=6200)
    parser.add_argument("--revise-with", type=int)
    parser.add_argument("--channel", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("Choose an output directory that does not exist")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = localize(args.config, args.input, args.concepts, args.layer,
                      args.step, args.device, args.revise_with)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    arrays = {key: result[key] for key in ("image", "probability", "raw_prediction", "concept_ids", "maps")}
    if "revision" in result:
        arrays.update(revision_grid=result["revision"]["grid"],
                      revision_mask=result["revision"]["mask"],
                      revision_activation=result["revision"]["activation"])
    np.savez_compressed(args.output_dir / "concepts.npz", **arrays)
    (args.output_dir / "metadata.json").write_text(json.dumps(dict(
        layer=args.layer, step=args.step, concept_ids=result["concept_ids"].tolist(),
        trained_inference_threshold=result["threshold"], maps_are_unscaled=True,
        reference_masks_used=False,
    ), indent=2) + "\n")
    figure = plot_concepts(result, args.channel)
    figure.savefig(args.output_dir / "concepts.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    if "revision" in result:
        figure = plot_revision(result, args.channel)
        figure.savefig(args.output_dir / "mask_revision.png", dpi=180, bbox_inches="tight")
        plt.close(figure)
    print(f"Saved concepts to {args.output_dir}")


if __name__ == "__main__":
    main()
