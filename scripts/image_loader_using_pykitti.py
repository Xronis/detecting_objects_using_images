import os
import time
import tempfile
import xml.etree.ElementTree

import pykitti
import numpy as np
import tensorflow as tf

from PIL import Image


def _load_images(basedir, date, drive):
    # The 'frames' argument is optional - default: None, which loads the whole dataset.
    # Calibration, timestamps, and IMU data are read automatically.
    # Camera and velodyne data are available via properties that create generators
    # when accessed, or through getter methods that provide random access.
    data = pykitti.raw(basedir, date, drive)

    # dataset.calib:         Calibration data are accessible as a named tuple
    # dataset.timestamps:    Timestamps are parsed into a list of datetime objects
    # dataset.oxts:          List of OXTS packets and 6-dof poses as named tuples
    # dataset.camN:          Returns a generator that loads individual images from camera N
    # dataset.get_camN(idx): Returns the image from camera N at idx
    # dataset.gray:          Returns a generator that loads monochrome stereo pairs (cam0, cam1)
    # dataset.get_gray(idx): Returns the monochrome stereo pair at idx
    # dataset.rgb:           Returns a generator that loads RGB stereo pairs (cam2, cam3)
    # dataset.get_rgb(idx):  Returns the RGB stereo pair at idx
    # dataset.velo:          Returns a generator that loads velodyne scans as [x,y,z,reflectance]
    # dataset.get_velo(idx): Returns the velodyne scan at idx

    return [image for image in data.cam2]


def load_images(date='2011_09_26', basedir='E:\Documents\KITTI\Raw'):
    folder = basedir+'\\{}'.format(date)
    drives = [name for name in os.listdir(folder) if os.path.isdir(folder+'\\{}'.format(name))]

    images = []

    for drive in drives:
        drive = drive.split('_')[-2]
        images += _load_images(basedir, date, drive)

    return images


def _transfort_images_to_npy_files(temp_file, basedir, date, drive):
    # The 'frames' argument is optional - default: None, which loads the whole dataset.
    # Calibration, timestamps, and IMU data are read automatically.
    # Camera and velodyne data are available via properties that create generators
    # when accessed, or through getter methods that provide random access.
    data = pykitti.raw(basedir, date, drive)

    # dataset.calib:         Calibration data are accessible as a named tuple
    # dataset.timestamps:    Timestamps are parsed into a list of datetime objects
    # dataset.oxts:          List of OXTS packets and 6-dof poses as named tuples
    # dataset.camN:          Returns a generator that loads individual images from camera N
    # dataset.get_camN(idx): Returns the image from camera N at idx
    # dataset.gray:          Returns a generator that loads monochrome stereo pairs (cam0, cam1)
    # dataset.get_gray(idx): Returns the monochrome stereo pair at idx
    # dataset.rgb:           Returns a generator that loads RGB stereo pairs (cam2, cam3)
    # dataset.get_rgb(idx):  Returns the RGB stereo pair at idx
    # dataset.velo:          Returns a generator that loads velodyne scans as [x,y,z,reflectance]
    # dataset.get_velo(idx): Returns the velodyne scan at idx

    temp_file_path = temp_file + '\{}'.format(np.random.randint(1, 1000000))

    for image in data.cam2:
        np.save(temp_file_path, np.array(image))


def transform_images(temp_file, date='2011_09_26', basedir='E:\Documents\KITTI\Raw'):
    folder = basedir+'\\{}'.format(date)
    drives = [name for name in os.listdir(folder) if os.path.isdir(folder+'\\{}'.format(name))]

    images = []

    for drive in drives:
        drive = drive.split('_')[-2]
        _transfort_images_to_npy_files(temp_file, basedir, date, drive)

    return images


def rotate_images(images):
    for i in range(len(images)):
        images[i] = images[i].rotate(2)

    return images

def load_labels(part, drive='2011_09_26', basedir='E:\Documents\KITTI\Raw'):


if __name__ == '__main__':

    start_time = time.time()

    basedir = 'C:\\Users\\ppanagiotidis\\Pictures\\Raw'
    date = '2011_09_26'

    parts = [name for name in os.listdir(basedir) if os.path.isdir(basedir+'\\{}'.format(name))]

    for part in parts:

        basedir_part_date = basedir + '\\' + part + '\\' + date
        basedir_part = basedir + '\\' + part

        drives = [name for name in os.listdir(basedir_part_date) if os.path.isdir(basedir_part_date + '\\{}'.format(name))]

        for drive in drives:
            drive = drive.split('_')[-2]
            # images = np.array(_load_images(basedir_part, date, drive)) CAST IMAGES TO ND.ARRAY
            images = rotate_images(_load_images(basedir_part, date, drive))

        # print(np.shape(images))

        images[0].show()

    print('Execution Time: {} secs'.format(time.time() - start_time))
