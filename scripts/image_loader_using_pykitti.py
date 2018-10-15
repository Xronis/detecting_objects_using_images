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


def load_images(drive, basedir='E:\Documents\KITTI\Raw', date='2011_09_26'):
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

    drives = os.listdir(basedir + '\\' + date + '\\')
    drives = [drive for drive in drives if os.path.isdir(basedir + '\\' + date + '\\' + drive)]

    for drive in drives:

        clean_drive = drive.split('_')[-2]
        images = load_images(basedir=basedir, date=date, drive=clean_drive)

        print('Loaded drive ' + clean_drive)
    # poses_of_frame = get_poses_of_frame(labels, 1)

    # for pose in poses_of_frame:
    #     pose.__str__()

    print('Execution Time: {} secs'.format(time.time() - start_time))
