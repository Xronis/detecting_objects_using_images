from glob import glob

import os

import cv2
import numpy as np


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

    try:
        images_path = np.array([image for folder in os.walk(path) for image in glob(os.path.join(folder[0], '*.jpg'))])

        images_array = np.array([])

        for image_path in images_path:
            images_array = np.append(images_array, cv2.imread(image_path))

        return images_array

    except Exception as e:
        print('Exception: {}'.format(e))
        exit(-1)


if __name__ == '__main__':
    print(image_loader('..\\photos_for_test\\'))
