"""Video timing, crop geometry and output safety without model assets."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from hand_kd import demo


class DemoTests(unittest.TestCase):
    def test_crop_predictions_return_to_source_pixels(self):
        affine = np.array([[.64, 0, 0], [0, .64, -12.8]], np.float32)
        uv = np.array([[.5, .5], [0, 0], [1, 1]], np.float32)
        np.testing.assert_allclose(demo.source_points(uv, affine),
                                   [[100, 120], [0, 20], [200, 220]], atol=1e-5)

    def test_variable_frame_rate_uses_previous_frame_and_keeps_gaps(self):
        # At 0.1 seconds the 0.04-second frame is current, not the future 0.2 one.
        times = np.array([0., .04, .2])
        np.testing.assert_array_equal(demo.display_indices(times, .3, 10), [0, 1, 2])
        np.testing.assert_array_equal(demo.display_indices(times, .4, 20),
                                      [0, 1, 1, 1, 2, 2, 2, 2])

    def test_detector_filters_hand_class_and_invalid_boxes(self):
        boxes = np.array([[0, 0, 10, 10], [20, 30, 70, 90], [50, 0, 40, 20]])
        scores = np.array([.99, .8, 1.])
        classes = np.array([0, 1, 1])
        np.testing.assert_array_equal(demo.select_box(boxes, scores, classes, 1, .5, 100, 100),
                                      [20, 30, 70, 90])
        self.assertIsNone(demo.select_box(boxes, scores, classes, 1, .9, 100, 100))

    def test_invalid_manual_box_is_rejected(self):
        for box in ([20, 0, 10, 20], [-1, 0, 10, 20], [0, 0, 101, 20], [0, 0, float('nan'), 20]):
            with self.subTest(box=box), self.assertRaises(ValueError):
                demo.validate_box(box, 100, 100)

    def test_publish_never_clobbers_user_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, dest = Path(tmp)/'new.mp4', Path(tmp)/'existing.mp4'
            source.write_bytes(b'new'); dest.write_bytes(b'original')
            with self.assertRaises(FileExistsError):
                demo.publish(source, dest)
            self.assertEqual(dest.read_bytes(), b'original')

    def test_missing_hand_frame_renders_without_reusing_predictions(self):
        frame = np.zeros((80, 120, 3), np.uint8)
        canvas = demo.render_frame(frame, None, None, None, 0.)
        self.assertEqual(canvas.shape, (640, 960, 3))
        # The lower right 3D panel contains no carried-over skeleton.
        self.assertTrue(np.all(canvas[380:620, 660:940] == 22))


if __name__ == '__main__':
    unittest.main()
