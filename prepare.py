#!/usr/bin/env python3
"""Prepare PI-CAI MRI slices and an input manifest."""

import argparse
import json
from pathlib import Path
import re

import h5py
import numpy as np
import SimpleITK as sitk

CHANNELS = ("t2w", "adc", "hbv")
SHAPE = (24, 384, 384)  # z, y, x
SPACING = (3.0, 0.5, 0.5)  # z, y, x, in mm


def input_cases(images_dir, labels_dir=None, cases=None, prepared=False):
    """Resolve explicit studies, or all studies with a first MRI sequence."""
    images_dir = Path(images_dir)
    if cases is None:
        candidates = images_dir.glob("*_0000.nii.gz" if prepared else "*/*_t2w.mha")
        suffix = "_0000.nii.gz" if prepared else "_t2w.mha"
        cases = sorted(path.name.removesuffix(suffix) for path in candidates)
    if not cases or len(set(cases)) != len(cases):
        raise ValueError("Supply distinct study IDs or an image directory containing studies")
    records = []
    for study in cases:
        if not re.fullmatch(r"\d{5}_\d{7}", study):
            raise ValueError(f"Expected a PI-CAI patient_study ID: {study}")
        patient = study.split("_")[0]
        scans = ([images_dir / f"{study}_{i:04d}.nii.gz" for i in range(3)] if prepared else
                 [images_dir / patient / f"{study}_{channel}.mha" for channel in CHANNELS])
        label = Path(labels_dir) / f"{study}.nii.gz" if labels_dir is not None else None
        for path in scans + ([label] if label is not None else []):
            if not path.is_file():
                raise FileNotFoundError(path)
        records.append((study, scans, label))
    return records


def geometry(image):
    return dict(size=image.GetSize(), spacing=image.GetSpacing(),
                origin=image.GetOrigin(), direction=image.GetDirection())


def prepare_case(scan_paths, label_path=None, prepared=False):
    """Use image geometry for resampling; labels never define the crop or grid."""
    scans = [sitk.ReadImage(str(path)) for path in scan_paths]
    label = sitk.ReadImage(str(label_path)) if label_path is not None else None
    for image in scans + ([label] if label is not None else []):
        if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
            raise ValueError("Each MRI sequence and annotation must be a scalar 3D volume")
        values = sitk.GetArrayViewFromImage(image)
        if not np.isfinite(values).all():
            raise ValueError("MRI sequences and annotations must contain finite values")
    if label is not None:
        values = sitk.GetArrayViewFromImage(label)
        if (values < 0).any() or not np.equal(values, np.floor(values)).all():
            raise ValueError("Supply nonnegative integer cancer lesion labels")
    if not prepared:
        from picai_prep.preprocessing import PreprocessingSettings, Sample

        sample = Sample(scans=scans, lbl=label,
                        settings=PreprocessingSettings(matrix_size=SHAPE, spacing=SPACING))
        sample.preprocess()
        scans, label = sample.scans, sample.lbl
    reference = geometry(scans[0])
    if reference["size"][::-1] != SHAPE or not np.allclose(reference["spacing"][::-1], SPACING, rtol=0, atol=1e-6):
        raise ValueError("Prepared volumes must have shape 24 x 384 x 384 and spacing 3 x 0.5 x 0.5 mm")
    for image in scans[1:] + ([label] if label is not None else []):
        current = geometry(image)
        if current["size"] != reference["size"] or any(
            not np.allclose(current[key], reference[key], rtol=0, atol=1e-5)
            for key in ("spacing", "origin", "direction")
        ):
            raise ValueError("Prepared MRI sequences and labels must share the same physical grid")
    # Store int16 images before prediction-time resizing.
    images = np.stack([sitk.GetArrayFromImage(image) for image in scans])
    if not np.isfinite(images).all() or images.min() < -32768 or images.max() > 32767:
        raise ValueError("Prepared MRI values must be finite and fit int16 storage")
    target = (sitk.GetArrayFromImage(label) > 0).astype(np.uint8) if label is not None else None
    return images.astype(np.int16), target


def prepare(images_dir, output_dir, labels_dir=None, cases=None, prepared=False):
    records = input_cases(images_dir, labels_dir, cases, prepared)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    samples = []
    for number, (study, scans, label) in enumerate(records, 1):
        images, target = prepare_case(scans, label, prepared)
        directory = output_dir / "slices" / study
        directory.mkdir(parents=True)
        for z in range(images.shape[1]):
            path = directory / f"{z}.hdf5"
            with h5py.File(path, "x") as output:
                output["ct"] = images[:, z]
                if target is not None:
                    output["seg"] = target[z]
            samples.append(dict(path=path.relative_to(output_dir).as_posix(),
                                patient_id=study.split("_")[0], study_id=study, slice_id=str(z)))
        print(f"Prepared examination {number}/{len(records)}: {study}", flush=True)
    manifest = dict(preprocessing=dict(channels=list(CHANNELS), shape_zyx=list(SHAPE),
                                     spacing_zyx_mm=list(SPACING), image_storage="int16",
                                     input_format="prepared_nifti" if prepared else "raw_mha"),
                    samples=samples)
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True,
                        help="Raw archive with patient/*.mha, or imagesTr/ with --prepared")
    parser.add_argument("--labels-dir", type=Path,
                        help="Optional directory of patient_study.nii.gz cancer annotations")
    parser.add_argument("--cases", nargs="+", help="Study IDs to prepare; default: all available studies")
    parser.add_argument("--prepared", action="store_true",
                        help="Read already aligned/cropped *_0000/0001/0002.nii.gz; skip resampling")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("Choose an output directory that does not exist")
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
    manifest = prepare(args.images_dir, args.output_dir, args.labels_dir, args.cases, args.prepared)
    print(f"Saved {manifest}")


if __name__ == "__main__":
    main()
