from glob import glob

import os

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

    images_path = np.array([image for folder in os.walk(path) for image in glob(os.path.join(folder[0], '*.jpg'))])

    images_dict = {}

    for i in range(len(images_path)):

        image = cv2.imread(images_path[i])
        image = image.flatten()
        images_dict['image_{}'.format(i)] = np.asarray(image)

    return images_dict


if __name__ == '__main__':
    images = image_loader('..\\photos_for_test\\')
    data_frame = pd.DataFrame.from_dict(images, orient='index')
    print(data_frame)
