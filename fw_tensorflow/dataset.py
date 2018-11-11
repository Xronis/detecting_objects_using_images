import numpy as np

from common.datasets import Dataset
from common.datareaders import KittiDataReader


class ThesisNetKittiDataset(Dataset):
    """Kitti dataset for the Thesis network.

    Attributes:
        kitti_reader (KittiDataReader): A Kitti dataset reader
        num_classes (int): The number of classes the network should categorize objects to
        stride (int): TODO add description of stride
        target_generator: TODO add description of target_generator
    """

    def __init__(self, data_readers, num_classes=None, stride=None):
        super(ThesisNetKittiDataset, self).__init__(data_readers)

        assert isinstance(data_readers[0], KittiDataReader)

        self.kitti_reader = data_readers[0]
        self.num_classes = num_classes
        self.stride = stride
        self.target_generator = None

    def init_readers(self):
        self.kitti_reader.read_filenames()
        self.target_generator = self.ThesisNetKittiTargetGenerator()
        self.target_generator.setup(num_classes=self.num_classes,
                                    image_height=self.kitti_reader.image_height,
                                    image_width=self.kitti_reader.image_width,
                                    stride=self.stride)

    def get(self, idxs, reader_idx=0):
        """Gets a batch of Kitti images
        Args:
            reader_idx (int, optional): Not used for a single reader
            idxs (list of int): A list of indices for a data batch
        Returns:
            images (ndarray): Kitti images, size (batch, height, width, channels)
            classes (ndarray): ThesisNet targets, size (batch, num_class, mask height, mask width)
            bboxes (ndarray): ThesisNet targets, size (batch, num_coords, mask height, mask width)
            cdepths (ndarray): ThesisNet targets, size (batch, num_depth, mask height, mask width)
            suppress (ndarray): ThesisNet suppress masks, size (batch, mask height, mask width)
        """
        images = None
        classes = None
        bboxes = None
        depths = None
        suppress = None
        if reader_idx < self.num_readers:
            kitti_images = []
            kitti_suppress = []
            kitti_classes = []
            kitti_bboxes = []
            kitti_depths = []
            for idx in idxs:
                data = self.kitti_reader.read(idx)
                image = data.images.color_left
                class_target, bbox_target, depth_target, suppress_mask = self.target_generator.generate(data.labels)
                kitti_images.append(image)
                kitti_classes.append(class_target)
                kitti_bboxes.append(bbox_target)
                kitti_depths.append(depth_target)
                kitti_suppress.append(suppress_mask)
            images = np.stack(kitti_images, axis=0)
            classes = np.stack(kitti_classes, axis=0)
            bboxes = np.stack(kitti_bboxes, axis=0)
            depths = np.stack(kitti_depths, axis=0)
            suppress = np.stack(kitti_suppress, axis=0)
        return images, classes, bboxes, depths, suppress
    class ThesisNetKittiTargetGenerator(object):
        """Thesis network target generator.
        Attributes:
            num_classes (int): The number of classes for classification
            stride (int): The model stride
            num_coords (int): The number of masks for coordinates (xmin, ymin, xmax, ymax)
            num_depth (int): The number of masks for depth (distance)
            target_height (int): The target/mask height
            target_width (int): The target/mask width
            classification_mask (ndarray): The classification mask
            bbox_mask (ndarray): The bounding box mask
            depth_mask (ndarray): The depth mask
            index_matrix_x (ndarray): Bounding box index matrix x
            index_matrix_y (ndarray): Bounding box index matrix y
        """
        def __init__(self):
            self.num_classes = None
            self.stride = 1
            self.num_coords = 4
            self.num_depth = 1
            self.target_height = None
            self.target_width = None
            self.classification_mask = None
            self.bbox_mask = None
            self.depth_mask = None
            self.index_matrix_x = None
            self.index_matrix_y = None
        def setup(self, num_classes=5, image_height=352, image_width=1216, stride=4):
            """Sets up the target generator.
            Initializes model information, target/mask size information and index matrices
            Args:
                num_classes (int, optional): The number of classes to account for
                image_height (int, optinoal): The (trimmed) input image height
                image_width (int, optional): The (trimmed) input image width
                stride (int, optional): The model stride
            """
            self.num_classes = num_classes
            self.stride = stride
            self.target_height = int(np.ceil(image_height / stride))
            self.target_width = int(np.ceil(image_width / stride))
            self.index_matrix_x = np.zeros([self.target_height, self.target_width])
            self.index_matrix_y = np.zeros([self.target_height, self.target_width])
            for i in range(self.target_height):
                self.index_matrix_x[i, :] = np.arange(0, image_width, self.stride)
            for i in range(self.target_width):
                self.index_matrix_y[:, i] = np.arange(0, image_height, self.stride)
        def generate(self, objects):
            """Generates a network target from the `objects`.
            Args:
                objects (KittiObject): Kitti object labels for a Kitti objects
            Returns:
                ndarray targets with classification, bounding box, depth and supression masks
            """
            # Initialize/reset each target volume with zeros
            self.classification_mask = np.zeros([self.num_classes, self.target_height, self.target_width])
            self.bbox_mask = np.zeros([self.num_coords, self.target_height, self.target_width])
            self.depth_mask = np.zeros([self.num_depth, self.target_height, self.target_width])
            # Set entire mask as background until objects are processed
            self.classification_mask[0, :, :] = 1
            for obj in objects:
                obj_class = self._get_class(obj)
                shrink_factor = self._get_shrink_factor(obj_class)
                mask_coords = self._get_mask_coords(obj, shrink_factor)
                self._update_classification_mask(obj_class, mask_coords)
                self._update_bbox_mask(obj.bounding_box, mask_coords)
                self._update_depth_mask(obj.bounding_box, mask_coords)
            # Suppress background and don't care classes
            suppress_mask = self._suppress_bg_dc()
            return np.copy(self.classification_mask), np.copy(self.bbox_mask), \
                   np.copy(self.depth_mask), np.copy(suppress_mask)
        def _suppress_bg_dc(self):
            """Suppresses background and don't care classes for the bounding box and depth masks.
            Returns:
                suppress_mask (ndarray): The suppress mask
            """
            # mask for suppressing background/don't care classes
            suppress_mask = 1 - (self.classification_mask[0] + self.classification_mask[1])
            # Suppress bounding box mask
            for i in range(self.num_coords):
                self.bbox_mask[i] = np.multiply(self.bbox_mask[i], suppress_mask)
            # Suppress for depth mask
            self.depth_mask = np.multiply(self.depth_mask, suppress_mask)
            return suppress_mask
        def _update_classification_mask(self, obj_class, mask_coords):
            """Updates the classification target/mask.
            Sets classification mask to 1 where object resides for the specified object class (dimension),
            removes (sets to 0) classification mask for background class
            Args:
                obj_class (int): Numeric representation of object class
                mask_coords (ndarray): Object bounding box coordinates [xmin, ymin, xmax, ymax] (target/mask coordinates)
            """
            # Remove background where object resides
            self.classification_mask[0, mask_coords[1]:mask_coords[3], mask_coords[0]:mask_coords[2]] = 0
            # Set classification mask where object resides
            self.classification_mask[obj_class, mask_coords[1]:mask_coords[3], mask_coords[0]:mask_coords[2]] = 1
        def _update_bbox_mask(self, bbox_coords, mask_coords):
            """Updates the bounding box mask.
            Sets the bounding box mask to the bounding box coordinate value where the object resides for each dim:
                dim 0: xmin
                dim 1: ymin
                dim 2: xmax
                dim 3: ymax
            Args:
                bbox_coords (ndarray): Object bounding box coordinates [xmin, ymin, xmax, ymax] (image coordinates)
                mask_coords (ndarray): Object bounding box coordinates [xmin, ymin, xmax, ymax] (target/mask coordinates)
            """
            # Set target/mask regions to image bounding box values
            self.bbox_mask[0, mask_coords[1]:mask_coords[3], mask_coords[0]:mask_coords[2]] = bbox_coords[0]
            self.bbox_mask[1, mask_coords[1]:mask_coords[3], mask_coords[0]:mask_coords[2]] = bbox_coords[1]
            self.bbox_mask[2, mask_coords[1]:mask_coords[3], mask_coords[0]:mask_coords[2]] = bbox_coords[2]
            self.bbox_mask[3, mask_coords[1]:mask_coords[3], mask_coords[0]:mask_coords[2]] = bbox_coords[3]
            # Encode bounding box mask
            self.bbox_mask[0, :] = (self.index_matrix_x - self.bbox_mask[0, :])
            self.bbox_mask[1, :] = (self.index_matrix_y - self.bbox_mask[1, :])
            self.bbox_mask[2, :] = (self.bbox_mask[2, :] - self.index_matrix_x)
            self.bbox_mask[3, :] = (self.bbox_mask[3, :] - self.index_matrix_y)
        def _update_depth_mask(self, bbox_coords, mask_coords):
            """Updates the depth target/mask.
            Sets depth mask to the object distance value where the object resides
            Args:
                bbox_coords (ndarray): Object bounding box coordinates [xmin, ymin, xmax, ymax] (image coordinates)
                mask_coords (ndarray): Object bounding box coordinates [xmin, ymin, xmax, ymax] (target/mask coordinates)
            """
            # Compute object distance in camera coordinates
            distance = np.linalg.norm(bbox_coords)
            # Set target/mask bounding box to distance value
            self.depth_mask[0, mask_coords[1]:mask_coords[3], mask_coords[0]:mask_coords[2]] = distance
        def _get_class(self, obj):
            """Get object class/type as integer value.
            Object classes are defined as:
                0: Background
                1: Don't care/Person sitting or highly occludes/truncated objects
                2: Car/Van
                3: Pedestrian
                4: Cyclist
            Args:
                obj (KittiObject): Kitti object labels for a Kitti object
            Returns:
                An integer representation of the object class
            """
            object_type = obj.object_type
            # Background class
            object_class = 0
            # Don't care classes
            if object_type in ['DontCare', 'Person_sitting'] or obj.truncation > 0.75 or obj.occlusion > 1:
                object_class = 1
            # Vehicle classes
            elif object_type in ['Car', 'Van']:
                object_class = 2
            # Pedestrian class
            elif object_type in ['Pedestrian']:
                object_class = 3
            # Cyclist class
            elif object_type in ['Cyclist']:
                object_class = 4
            return object_class
        def _get_shrink_factor(self, obj_class):
            """Get shrink factor 0.2 for don't care classes, otherwise 0.5.
            Args:
                obj_class (int): Integer representation of object class
            Returns:
                The shrink factor
            """
            return 0.5 if obj_class == 1 else 0.2
        def _get_mask_coords(self, obj, shrink_factor):
            """Get bounding box coordinates mapped to target/mask coordinates.
            Args:
                obj (KittiObject): Kitti object labels for a Kitti object
                shrink_factor (float): A shrink factor from image to target/mask coordinates
            Returns:
                The bounding box coordinates in the target/mask coordinate system.
            """
            # split bounding box into x,y coordinates
            xmin, ymin, xmax, ymax = np.split(obj.bounding_box, len(obj.bounding_box))
            # compute bounding box center coordinate
            bbox_center = np.array([xmin + (xmax - xmin) / 2, ymin + (ymax - ymin) / 2])
            # compute target/mask center coordinate
            mask_center = bbox_center / self.stride
            # compute the shrinked dimensions (? check this)
            shrink_dim = np.array([(xmax - xmin) * np.sqrt(shrink_factor),
                                   (ymax - ymin) * np.sqrt(shrink_factor)]) / self.stride
            # compute x,y target/mask coordinates
            mask_coords = np.array([np.max([0, np.min([self.target_width, (mask_center[0] - shrink_dim[0] / 2)])]),
                                    np.max([0, np.min([self.target_height, (mask_center[1] - shrink_dim[1] / 2)])]),
                                    np.max([0, np.min([self.target_width, (mask_center[0] + shrink_dim[0] / 2)])]),
                                    np.max([0, np.min([self.target_height, (mask_center[1] + shrink_dim[1] / 2)])])],
                                   dtype=np.int32)
            return mask_coords