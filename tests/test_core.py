"""Asset-free checks of the released experiment contract."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from hand_kd.data import student_crop
from hand_kd.evaluation import summarize_errors
from hand_kd.losses import feature_loss
from experiments.single_stack.analyze import validate_completion_summary
from hand_kd.train import epoch_plan


ROOT = Path(__file__).resolve().parents[1]


class ContractTests(unittest.TestCase):
    def test_paired_epoch_budget(self):
        full, omitted = epoch_plan(np.arange(237517), 20260929, 0, 32)
        self.assertEqual((len(full), len(omitted)), (237504, 13))
        np.testing.assert_array_equal(np.sort(np.r_[full, omitted]), np.arange(237517))

    def test_feature_teacher_is_detached(self):
        prediction = torch.randn(2, 4, 3, 3, requires_grad=True)
        teacher = torch.randn_like(prediction, requires_grad=True)
        feature_loss(prediction, teacher).backward()
        self.assertIsNotNone(prediction.grad)
        self.assertIsNone(teacher.grad)

    def test_crop_geometry(self):
        crop, affine = student_crop(np.zeros((240, 320, 3), np.uint8), [50, 70, 150, 170])
        self.assertEqual(crop.shape, (3, 128, 128))
        np.testing.assert_allclose(affine @ np.array([100., 120., 1.]), [64., 64.])

    def test_root_relative_metric(self):
        truth = np.zeros((2, 21, 3), np.float32)
        truth[:, 1, 0] = .1
        prediction = truth + np.array([1., 2., 3.], np.float32)
        metrics = summarize_errors(prediction, truth)
        self.assertLess(metrics['mpjpe_mm'], 1e-4)
        self.assertLess(metrics['pa_mpjpe_mm'], 1e-4)

    def test_report_rejects_summary_mismatch(self):
        rows = [dict(epoch=i + 1, arms={a: dict(metrics={'mpjpe_mm': 40. - i})
                 for a in ('A', 'B')}) for i in range(38)]
        summary = dict(complete=True, epochs=38, updates_per_arm=282036,
                       images_per_arm=9025152, augmented_views_per_arm=18050304,
                       final={a: deepcopy(rows[-1]['arms'][a]['metrics']) for a in ('A', 'B')},
                       best_development_epoch={'A': 38, 'B': 38})
        self.assertEqual(validate_completion_summary(summary, rows)[0], summary['final'])
        summary['final']['B']['mpjpe_mm'] = -1
        with self.assertRaises(ValueError):
            validate_completion_summary(summary, rows)

    def test_published_summary_matches_epoch_38(self):
        rows = json.loads((ROOT/'experiments/single_stack/results/epoch-metrics.json').read_text())
        receipt = json.loads((ROOT/'experiments/single_stack/results/verified-metrics.json').read_text())
        self.assertEqual(len(rows), 38)
        for arm in ('A', 'B'):
            self.assertAlmostEqual(rows[-1][arm]['metrics']['mpjpe_mm'],
                                   receipt['final'][arm]['mpjpe_mm'], places=5)


if __name__ == '__main__':
    unittest.main()
