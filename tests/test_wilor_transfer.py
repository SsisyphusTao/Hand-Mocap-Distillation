"""Asset-free checks for WiLoR feature alignment and microbatch weighting."""
import unittest

import torch

from experiments.wilor_transfer.features import FeatureAligner, feature_loss, student_regions
from experiments.wilor_transfer.support import micro_slices



class WiLoRTransferTests(unittest.TestCase):
    def test_feature_alignment_keeps_teacher_frozen_and_reaches_each_student_stage(self):
        torch.manual_seed(3)
        stages = [torch.randn(2, c, h, w, requires_grad=True) for c, h, w in
                  ((128, 8, 8), (128, 8, 8), (24, 8, 8), (256, 8, 8))]
        teacher = torch.randn(2, 1280, 8, 6, requires_grad=True)
        for mode in ('projected', 'region'):
            for stage in stages:
                stage.grad = None
            aligner = FeatureAligner(mode=mode)
            prediction = aligner(stages)
            self.assertEqual(prediction.shape, teacher.shape)
            feature_loss(prediction, teacher).backward()
            self.assertTrue(all(stage.grad is not None and stage.grad.abs().sum() > 0
                                for stage in stages))
            self.assertIsNone(teacher.grad)

    def test_student_regions_use_central_three_quarters_of_width(self):
        width = torch.arange(8, dtype=torch.float32).view(1, 1, 1, 8).expand(1, 1, 4, 8)
        regions = student_regions(width, grid=(2, 2))
        self.assertEqual(regions.shape, (1, 1, 2, 2))
        torch.testing.assert_close(regions[0, 0, 0], torch.tensor([2., 5.]))

    def test_tail_microbatch_weights_are_sample_weighted(self):
        batches = list(micro_slices(13, 8))
        self.assertEqual([(s.start, s.stop) for s, _ in batches], [(0, 8), (8, 13)])
        self.assertAlmostEqual(sum(weight for _, weight in batches), 1.)
        self.assertAlmostEqual(sum(weight * value for (_, weight), value in
                                   zip(batches, (1., 3.))), 23 / 13)



if __name__ == '__main__':
    unittest.main()
