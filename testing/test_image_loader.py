from glob import glob
from scripts.image_loader import image_loader as loader

import unittest
import os

import numpy as np
import cv2


class TestImageLoader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def setUp(self):
        self.path = '..\\photos_for_test\\'

    def test_image_loader_appropriate_values(self):
        images_path = np.array([image for folder in os.walk(self.path) for image in glob(os.path.join(folder[0], '*.jpg'))])

        expected_images_array = np.array([])

        for image_path in images_path:
            expected_images_array = np.append(expected_images_array, cv2.imread(image_path))

        actual_images_array = loader(self.path)

        self.assertEqual(expected_images_array.all(), actual_images_array.all())


if __name__ == '__main__':
    unittest.main(verbosity=2)
