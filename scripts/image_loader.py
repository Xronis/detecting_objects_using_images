from glob import glob

import os
import time

import cv2
import numpy as np
import pandas as pd


class BadInputError(ValueError):
    pass


class NotValidPathError(ValueError):
    pass


def image_loader(path):
    """
    Returns an np.array that contains the images in the given path. If the :param recursive is True,
    the function will do a recursive search for images inside the folder structure of the path.

    If there are no images in that path, it returns an empty array.
    If the path is not in a valid format, it raises a BadInputError.

    :param path: str, Folder path that contains images intended to be loaded.

    :return: An array of the loaded images.
    """

    if type(path) is not str:
        raise BadInputError('BadInputError: Type {} is not supported. '
                            'Path should always be a str'.format(type(path)))

    if not os.path.exists(path=path):
        raise NotValidPathError('NotValidPathError: The given path does not exist.')

    images_path = np.array([image for folder in os.walk(path) for image in glob(os.path.join(folder[0], '*.png'))])

    images_dict = {}

    for i in range(len(images_path)):

        image = cv2.imread(images_path[i])
        image = image.flatten()
        image_name = images_path[i].split('\\')[-1].split('.')[0]

        images_dict[image_name] = np.asarray(image)
        print('Loading image {}/{}'.format(i+1, len(images_path)), end='\r')

    return images_dict


if __name__ == '__main__':

    start_time = time.time()
    images = image_loader('E:\Documents\KITTI\Images\\training\image_2')
    print('Execution time: {} secs'.format(time.time() - start_time))
