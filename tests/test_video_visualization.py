"""Unit and static regression tests for demo_video's output composition."""

import pathlib
import unittest

import cv2
import numpy as np
from skimage.transform import estimate_transform, warp

from utils.video_visualization import (
    compose_overlay_layout,
    compose_render_overlay,
    rgb_to_opencv_bgr,
)


class TestOverlayComposition(unittest.TestCase):
    def setUp(self):
        self.original = np.full((3, 4, 3), [10, 20, 30], dtype=np.uint8)
        self.rendered = np.zeros_like(self.original)
        self.rendered[1, 2] = [255, 255, 255]
        self.mask = (self.rendered != 0).any(axis=2)
        self.color = (70, 190, 255)

    def test_three_panel_shape_content_and_mask(self):
        layout = compose_overlay_layout(
            self.original, self.rendered, self.mask, 0.5, self.color)
        self.assertEqual(layout.shape, (3, 12, 3))
        np.testing.assert_array_equal(layout[:, :4], self.original)
        np.testing.assert_array_equal(layout[:, 8:], self.rendered)
        np.testing.assert_array_equal(layout[0, 4:8], self.original[0])
        np.testing.assert_array_equal(layout[1, 6], [40, 105, 142])

    def test_alpha_extremes(self):
        zero = compose_render_overlay(
            self.original, self.rendered, self.mask, 0, self.color)
        one = compose_render_overlay(
            self.original, self.rendered, self.mask, 1, self.color)
        np.testing.assert_array_equal(zero, self.original)
        np.testing.assert_array_equal(one[1, 2], self.color)
        np.testing.assert_array_equal(one[0, 0], self.original[0, 0])

    def test_render_intensity_preserves_shading(self):
        rendered = self.rendered.copy()
        rendered[1, 1] = [64, 64, 64]
        mask = (rendered != 0).any(axis=2)
        result = compose_render_overlay(self.original, rendered, mask, 1, self.color)
        self.assertTrue(np.all(result[1, 1] < result[1, 2]))

    def test_crop_transform_can_restore_original_coordinates(self):
        small = np.zeros((2, 2, 3), dtype=np.uint8)
        small[0, 0] = 255
        mask = (small != 0).any(axis=2)
        identity = estimate_transform(
            'similarity', np.array([[0, 0], [0, 1], [1, 0]]),
            np.array([[0, 0], [0, 1], [1, 0]]))
        restored = warp(small, identity, output_shape=(4, 5), preserve_range=True).astype(np.uint8)
        restored_mask = warp(mask.astype(np.uint8), identity, output_shape=(4, 5),
                             order=0, preserve_range=True).astype(bool)
        original = np.zeros((4, 5, 3), dtype=np.uint8)
        layout = compose_overlay_layout(original, restored, restored_mask)
        self.assertEqual(layout.shape, (4, 15, 3))
        np.testing.assert_array_equal(layout[:, 10:], restored)

    def test_opencv_output_is_bgr(self):
        rgb = np.array([[[1, 2, 3]]], dtype=np.uint8)
        np.testing.assert_array_equal(rgb_to_opencv_bgr(rgb), [[[3, 2, 1]]])

    def test_legacy_path_remains_separate_and_unchanged(self):
        source = (pathlib.Path(__file__).parents[1] / 'demo_video.py').read_text()
        self.assertIn("if args.visualization_layout == 'overlay':", source)
        self.assertIn('grid = torch.cat([full_image, rendered_img_orig], dim=3)', source)
        self.assertIn('grid = torch.cat([cropped_image_tensor, rendered_img], dim=3)', source)
        self.assertIn('elif args.use_smirk_generator:\n        out_width *= 3', source)


if __name__ == '__main__':
    unittest.main()
