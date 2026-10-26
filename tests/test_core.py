import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import SimpleITK as sitk
import torch

from prepare import input_cases, prepare, prepare_case
from medfoe.baselines import predict_mask, slice_scores
from medfoe.concepts import notebook_activation, revise_mask
from medfoe.data import examinations, metrics, read_image
from medfoe.postprocess import calibrated_keep, rescue_empty_slices
from medfoe.sae import SAE, normalize_embeddings


class CoreTests(unittest.TestCase):
    def test_normalization_handles_zero_tokens(self):
        values = torch.tensor([[0., 0.], [3., 4.]])
        expected = torch.tensor([[0., 0.], [.6, .8]])
        torch.testing.assert_close(normalize_embeddings(values), expected)

    def test_saved_threshold_does_not_force_eight_features(self):
        state = {
            "encoder.weight": torch.zeros(1536, 768), "encoder.bias": torch.ones(1536),
            "decoder.weight": torch.zeros(768, 1536), "b_dec": torch.zeros(768),
            "k": torch.tensor(8), "threshold": torch.tensor(.5),
        }
        sae = SAE(state)
        self.assertEqual(int((sae.encode(torch.zeros(1, 768)) > 0).sum()), 1536)

    def test_image_reader_does_not_require_annotations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.hdf5"
            with h5py.File(path, "w") as source:
                source["ct"] = np.stack([np.ones((4, 5)), np.zeros((4, 5)), -np.ones((4, 5))])
            image = read_image(path)
            self.assertEqual(tuple(image.shape), (3, 1024, 1024))
            self.assertTrue(torch.isfinite(image).all())
            self.assertEqual(float(image[1].sum()), 0)

    def test_complete_examinations_reject_gaps_and_patient_mixing(self):
        rows = [dict(patient_id="a", study_id="exam", slice_id=str(i)) for i in range(3)]
        self.assertEqual(examinations(rows), [[0, 1, 2]])
        with self.assertRaises(ValueError):
            examinations(rows[1:])
        rows[1]["patient_id"] = "b"
        with self.assertRaises(ValueError):
            examinations(rows)

    def test_fallback_preserves_nonempty_primary_slices(self):
        keep, _ = rescue_empty_slices(
            np.array([True, False, False]), np.array([False, True, True]),
            np.array([4, 8, 3]), np.array([0, 0, 1]), 2,
        )
        np.testing.assert_array_equal(keep, [True, False, True])

    def test_configured_concept_counts_supply_spatial_seed_support(self):
        import json

        config = Path(__file__).resolve().parents[1] / "configs/method.json"
        point = json.loads(config.read_text())["reference"]
        inputs = dict(patch=np.full(2, .4), image=np.full(2, .9),
                      area=np.full(2, 256), patch_slice=np.arange(2),
                      active_concepts=np.full(2, 310), active_l11=np.zeros(2))
        topology = dict(component_of_patch=np.zeros(2, int))
        self.assertTrue(calibrated_keep(inputs, point, topology).all())
        disabled = dict(point, concept_count_policy="disabled")
        self.assertFalse(calibrated_keep(inputs, disabled, topology).any())

    def test_baselines_use_prediction_scores_and_return_actual_masks(self):
        logits = np.full((4, 4), np.log(9))
        scores = slice_scores(logits)
        self.assertAlmostEqual(scores[0], .1)
        self.assertAlmostEqual(scores[3], -np.log(10))
        native = dict(minimum_component_pixels=0, boundary_offset_pixels=0)
        self.assertTrue(predict_mask(logits, "max_prob", .11, native).all())
        self.assertFalse(predict_mask(logits, "max_prob", .09, native).any())

    def test_metrics_include_empty_outputs_on_positive_slices(self):
        result = metrics([[4, 4, 4], [5, 0, 0], [0, 2, 0]])
        self.assertEqual(result["f1"], .5)
        self.assertEqual(result["positive_slice_dice"], .5)
        self.assertIsNone(metrics([[0, 0, 0]])["positive_slice_dice"])

    def test_concept_reconstruction_needs_positive_evidence(self):
        codes = np.zeros((2, 2, 32), np.float32)
        raw = np.ones((8, 8), bool)
        self.assertFalse(revise_mask(codes, raw, 0)["mask"].any())
        self.assertFalse(notebook_activation(np.zeros((1, 2, 2)), (8, 8)).any())


class PreparationTests(unittest.TestCase):
    def setUp(self):
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)

    def write_volume(self, path, values, origin=(0., 0., 0.)):
        image = sitk.GetImageFromArray(values)
        image.SetSpacing((.5, .5, 3.))
        image.SetOrigin(origin)
        sitk.WriteImage(image, str(path))

    def test_raw_preparation_keeps_all_slices_without_annotations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images" / "99999"
            images.mkdir(parents=True)
            for value, channel in enumerate(("t2w", "adc", "hbv"), 1):
                self.write_volume(images / f"99999_9999999_{channel}.mha",
                                  np.full((3, 12, 16), value + .25, np.float32))
            manifest = prepare(root / "images", root / "prepared")
            from medfoe.data import read_manifest
            rows = read_manifest(manifest)["samples"]
            self.assertEqual(len(rows), 24)
            self.assertEqual(examinations(rows), [list(range(24))])
            with h5py.File(rows[11]["path"]) as output:
                self.assertNotIn("seg", output)
                self.assertEqual(output["ct"].dtype, np.dtype("int16"))
                np.testing.assert_array_equal(output["ct"][:, 192, 192], [1, 2, 3])
            with h5py.File(rows[0]["path"]) as output:
                self.assertFalse(np.asarray(output["ct"]).any())

    def test_annotations_do_not_change_image_preparation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scans = []
            for value in range(3):
                path = root / f"{value}.mha"
                self.write_volume(path, np.full((3, 12, 16), value + 1, np.int16))
                scans.append(path)
            label = np.zeros((3, 12, 16), np.uint8)
            label[1, 6, 8] = 3
            self.write_volume(root / "lesion.nii.gz", label)
            unlabeled, target = prepare_case(scans)
            labeled, mask = prepare_case(scans, root / "lesion.nii.gz")
            self.assertIsNone(target)
            np.testing.assert_array_equal(unlabeled, labeled)
            self.assertEqual(mask.dtype, np.dtype("uint8"))
            self.assertEqual(int(mask.sum()), 1)
            self.assertEqual(int(mask[11, 192, 192]), 1)

    def test_prepared_volumes_preserve_values_and_reject_grid_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scans = []
            values = np.arange(24, dtype=np.float32)[:, None, None] + np.zeros((24, 384, 384), np.float32)
            for channel in range(3):
                path = root / f"99999_9999999_{channel:04d}.nii.gz"
                self.write_volume(path, values + channel * 10 + .75)
                scans.append(path)
            with self.assertRaises(FileNotFoundError):
                input_cases(root, labels_dir=root, prepared=True)
            label = np.zeros((24, 384, 384), np.uint8)
            label[7, 10, 12] = 2
            label_path = root / "99999_9999999.nii.gz"
            self.write_volume(label_path, label)
            images, target = prepare_case(scans, label_path, prepared=True)
            np.testing.assert_array_equal(images[:, :, 0, 0], np.arange(24)[None] + 10 * np.arange(3)[:, None])
            self.assertEqual(int(target.sum()), 1)
            self.write_volume(scans[1], values, origin=(1., 0., 0.))
            with self.assertRaisesRegex(ValueError, "physical grid"):
                prepare_case(scans, label_path, prepared=True)
            scans[1].unlink()
            with self.assertRaises(FileNotFoundError):
                input_cases(root, prepared=True)


if __name__ == "__main__":
    unittest.main()
