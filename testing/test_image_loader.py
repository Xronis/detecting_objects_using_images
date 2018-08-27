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
        images_path = np.array([image for folder in os.walk(path) for image in glob(os.path.join(folder[0], '*.png'))])

        expected_images_dict = {}

        for i in range(len(images_path)):
            image = cv2.imread(images_path[i])
            image = image.flatten()
            image_name = images_path[i].split('\\')[-1].split('.')[0]

            expected_images_dict[image_name] = np.asarray(image)

        actual_images_dict = loader(self.path)

        self.assertDictEqual(expected_images_dict, actual_images_dict)


if __name__ == '__main__':
    unittest.main(verbosity=2)
