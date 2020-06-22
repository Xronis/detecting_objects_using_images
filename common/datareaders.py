import os
import numpy as np
from scipy.misc import imread
from abc import ABCMeta
from abc import abstractmethod
from common.dataencoders import KittiData


class DataReader(metaclass=ABCMeta):
    """Data reader interface."""

    @abstractmethod
    def read(self, idx):
        """Should read a data point from a dataset.
        Returns:
            A data point from a dataset
        """

    @abstractmethod
    def read_filenames(self):
        """Should initialize the data reader."""

    @property
    @abstractmethod
    def size(self):
        """Should provide the size of a dataset.
        Returns:
            The dataset size
        """


class KittiDataReader(DataReader):
    """Kitti data point reader.

    Reads image, label and calibration files of the Kitti dataset and provides KittiData objects

    Attributes:
        dataset_path (str): Base directory path to Kitti dataset
        color_left_filenames (list of str): A list of left color image file names
        color_right_filenames (list of str): A list of right color image file names
        label_filenames (list of str): A list of label file names
        calib_filenames (list of str): A list of calib file names
        read_color_left (bool, optional): True if left color image should be read
        read_color_right (bool, optional): True if right color image should be read
        read_labels (bool, optional): True if labels should be read
        read_calib (bool, optional): True if calibration data should be read
        image_width (int): Pixel width of camera image
        image_height (int): Pixel height of camera image
        image_channels (int): Number of color channels (for color images)
        num_images (int): The number of images in the dataset
        mean (list of floats): List of means for the RGB data channels
        std (list of floats): List of stds for the RGB data channels
    """
    def __init__(self, dataset_path, read_color_left=True, read_color_right=True, read_labels=True, read_calib=True):
        super(KittiDataReader, self).__init__()

        self.dataset_path = dataset_path
        self.color_left_filenames = None
        self.color_right_filenames = None
        self.label_filenames = None
        self.calib_filenames = None
        self.read_color_left = read_color_left
        self.read_color_right = read_color_right
        self.read_labels = read_labels
        self.read_calib = read_calib
        self.image_width = 1216
        self.image_height = 352
        self.image_channels = 3
        self.num_images = 7398
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    @property
    def size(self):
        """Returns the size of the dataset"""
        return self.num_images

    def read(self, idx=None):
        """Reads a Kitti data point with index `idx`.
        Args:
            idx (int): A data point index ranging [0,7398]
        Returns:
            A KittiData object
        """

        "Initialize variables to None"
        color_left = None
        color_right = None
        labels = None
        calib = None

        "Checks if read_color_left is enabled and if it is, calls private function _read_color_left"
        if self.read_color_left:
            color_left = self._read_color_left(idx)

        "Checks if read_color_right is enabled and if it is, calls private function _read_color_right"
        if self.read_color_right:
            color_right = self._read_color_right(idx)

        "Checks if read_labels is enabled and if it is, calls private function _read_labels"
        if self.read_labels:
            labels = self._read_labels(idx)

        "Checks if _read_calib is enabled and if it is, calls private function _read_calib"
        if self.read_calib:
            calib = self._read_calib(idx)

        images = KittiData.KittiImages(color_left=color_left, color_right=color_right)
        return KittiData(idx=idx, images=images, labels=labels, calib=calib)

    def read_filenames(self):
        """Get all data filenames."""

        self.color_left_filenames = self._read_sorted_filenames(
            os.path.join(self.dataset_path, 'object/training/image_2'), '.png')

        self.color_right_filenames = self._read_sorted_filenames(
            os.path.join(self.dataset_path, 'object/training/image_3'), '.png')

        self.label_filenames = self._read_sorted_filenames(
            os.path.join(self.dataset_path, 'object/training/label_2'), '.txt')

        self.calib_filenames = self._read_sorted_filenames(
            os.path.join(self.dataset_path, 'object/training/calib'), '.txt')

    def _read_sorted_filenames(self, path, endswith):
        """Get all filenames contained in path in a sorted array.
        Filenames are restricted to ones ending with a certain string.

        Args:
            path (str): The path containing the files who's names we want to find out
            endswith (str): Ending of files we are inderested in

        Returns:
            A sorted list of the filenames with the specified ending we are looking for
        """
        return sorted([os.path.join(path, file) for file in os.listdir(path) if file.endswith(endswith)])

    def _read_color_left(self, idx):
        """Reads a Kitti image from left color camera.
        Args:
            idx (int): A data point index ranging [0, self.num_images]
        Returns:
            The idx'th Kitti image from the left color camera
        Raises:
            IndexError is `idx` is out of range
        """
        image = imread(self.color_left_filenames[idx])
        return self._trim_image(self._normalize_image(image, mean=self.mean, std=self.std))

    def _read_color_right(self, idx):
        """Reads a Kitti image from right color camera.
        Args:
            idx (int): A data point index ranging [0, self.num_images]
        Returns:
            The idx'th Kitti image from the right color camera
        Raises:
            IndexError is `idx` is out of range
        """
        image = imread(self.color_right_filenames[idx])
        return self._trim_image(self._normalize_image(image, mean=self.mean, std=self.std))

    def _read_labels(self, idx):
        """Reads all object labels for the idx'th Kitti image.
        Args:
            idx (int): A data point index ranging [0, self.num_images]
        Returns:
            A KittiObject for the idx'th Kitti data point
        Raises:
            IndexError is `idx` is out of range
        """
        object_count = 0
        object_list = []

        "Open idx'th label file"
        with open(self.label_filenames[idx]) as object_labels:
            for line in object_labels:
                labels = line.split()

                "Create a new instance of the KittiObject class and append it to the object list"
                object_list.append(
                    KittiData.KittiObject(object_idx=object_count, object_type=labels[0],
                                          truncation=np.array(labels[1], dtype=np.float32),
                                          occlusion=np.array(labels[2], dtype=np.int32),
                                          alpha=np.array(labels[3], dtype=np.float32),
                                          bounding_box=np.array(labels[4:8], dtype=np.float32),
                                          dimensions=np.array(labels[8:11], dtype=np.float32),
                                          location=np.array(labels[11:14], dtype=np.float32),
                                          rotation=np.array(labels[14], dtype=np.float32)))
                object_count += 1

        return object_list

    def _read_calib(self, idx):
        """Read Kitti calibration data of the idx'th data point.
        Args:
            idx (int): A data point index ranging [0, self.num_images]
        Returns:
            A KittiCalibration object for the idx'th Kitti data point
        Raises:
            IndexError is `idx` is out of range
        """
        params = []

        "Read idx'th calib file"
        with open(self.calib_filenames[idx]) as calib:

            for line in calib:
                "Split first at ':' and then at ' '"
                params.append(np.array(line.split(sep=':')[-1].split(), dtype=np.float32))

        "Reshape params individualy to a 3x4 matrix"
        p0 = np.reshape(params[0], (3, 4))
        p1 = np.reshape(params[1], (3, 4))
        p2 = np.reshape(params[2], (3, 4))
        p3 = np.reshape(params[3], (3, 4))

        "Reshape r0_rect to a 4x4 matrix"
        r0_3x3 = np.reshape(params[4], (3, 3))
        r0_4x4 = np.hstack((np.vstack((r0_3x3, np.array([0, 0, 0]))), np.array([[0], [0], [0], [1]])))

        "Reshape tr_vel_to_cam to a 4x4 matrix"
        tr_vel_3x3 = np.reshape(params[5], (3, 4))
        tr_vel_4x4 = np.vstack((tr_vel_3x3, np.array([0, 0, 0, 1])))

        "Reshape tr_imu_to_vel to a 4x4 matrix"
        tr_imu_3x3 = np.reshape(params[5], (3, 4))
        tr_imu_4x4 = np.vstack((tr_imu_3x3, np.array([0, 0, 0, 1])))

        return KittiData.KittiCalibration(p0=p0, p1=p1, p2=p2, p3=p3,
                                          r0_rect=r0_4x4,
                                          tr_vel_to_cam=tr_vel_4x4,
                                          tr_imu_to_vel=tr_imu_4x4)

    def _trim_image(self, img):
        """Trims input image according to size specified.
        Trims pixels from the right and bottom edges of the image to keep consistency w.r.t.
        object label information
        Args:
            img (ndarray): Input image
        Returns:
            The trimmed image
        """

        "Check if the image is larger than the given width and height"
        if self.image_width < img.shape[1] and self.image_height < img.shape[0]:
            trimmed_image = img[0:self.image_height, 0:self.image_width, :]

        else:
            trimmed_image = img

        return trimmed_image

    def _normalize_image(self, img, mean=None, std=None):
        """Normalizes the input image by subtracting mean and dividing by std.
        Args:
            img (ndarray): input image as numpy array
            mean (list of float): mean to subtracted provided as list corresponding to RGB channels
            std (list of float): std to divide the RGB channels
        Returns:
            The normalized image
        """

        img = img / 255

        img[:, :, 0] = (img[:, :, 0] - mean[0]) / std[0]
        img[:, :, 1] = (img[:, :, 1] - mean[1]) / std[1]
        img[:, :, 2] = (img[:, :, 2] - mean[2]) / std[2]

        return img
