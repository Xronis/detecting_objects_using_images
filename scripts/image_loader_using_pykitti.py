import os
import sys
import time
import tempfile
import xml.etree.ElementTree

import pykitti
import numpy as np
import tensorflow as tf

from PIL import Image

from scripts.label_loader import load_labels


def _load_images(drive, basedir='E:\Documents\KITTI\Raw', date='2011_09_26'):
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


def _load_labels(basedir_part_date):
    drives = [name for name in os.listdir(basedir_part_date) if os.path.isdir(basedir_part_date+'\\'+name)]
    tracklet_xml_files = list(map(lambda x: basedir_part_date+'\\'+x+'\\'+'tracklet_labels.xml', drives))

    objects_of_file = []

    for xml_file in tracklet_xml_files:
        objects_of_file.extend(load_labels(xml_file))

    return objects_of_file


def _load_images_wrapper(basedir_part, date, drives):
    images = []

    for drive in drives:
        drive = drive.split('_')[-2]
        # images = np.array(_load_images(basedir_part, date, drive)) CAST IMAGES TO ND.ARRAY
        # images = rotate_images(_load_images(basedir_part, date, drive))
        images.extend(_load_images(basedir=basedir_part, date=date, drive=drive))
        print('Loaded drive '+drive)

    return images


def get_poses_of_frame(labels, frame):
    poses_of_frame = []

    for label in labels:
        pose = label.get_pose_at_frame(frame)
        if pose:
            poses_of_frame.append(pose)

    return poses_of_frame


if __name__ == '__main__':

    start_time = time.time()

    basedir = 'E:\Documents\KITTI\Raw'
    date = '2011_09_26'

    parts = [name for name in os.listdir(basedir) if os.path.isdir(basedir+'\\{}'.format(name))]

    for part in parts:

        basedir_part_date = basedir + '\\' + part + '\\' + date
        basedir_part = basedir + '\\' + part

        drives = [name for name in os.listdir(basedir_part_date) if os.path.isdir(basedir_part_date + '\\{}'.format(name))]

        images = _load_images_wrapper(basedir_part, date, drives)
        labels = _load_labels(basedir_part_date)

        poses_of_frame = get_poses_of_frame(labels, 1)

        for pose in poses_of_frame:
            pose.__str__()

    print('Execution Time: {} secs'.format(time.time() - start_time))
