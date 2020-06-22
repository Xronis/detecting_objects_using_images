class KittiData:
    """Kitti data point object.
    Attributes:
        idx (int): Data point index
        images (KittiImages): Images from left/right, gray/color cameras
        labels (KittiObject): Labels for kitti objects
        calib (KittiCalibration): Calibration data
    """

    def __init__(self, idx=None, images=None, labels=None, calib=None):
        self.idx = idx
        self.images = images
        self.labels = labels
        self.calib = calib

    class KittiImages:
        """Kitti data point camera images.
        Attributes:
            gray_left (ndarray): Left gray scale image
            gray_right (ndarray): Right gray scale image
            color_left (ndarray): Left color image
            color_right (ndarray): Right color image
        """

        def __init__(self, gray_left=None, gray_right=None, color_left=None, color_right=None):
            self.gray_left = gray_left
            self.gray_right = gray_right
            self.color_left = color_left
            self.color_right = color_right

    class KittiObject:
        """Kitti data point object label data.

        The Attributes of a single object in a Kitti data point.

        Attributes:
            object_idx (int): Index for this object in the data point

            object_type (str): Type of object, 'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting',
                'Cyclist', 'Tram', 'Misc' or 'DontCare'

            truncation (float): From 0 (non-truncated) to 1 (truncated), where truncated refers to the
                object leaving image boundaries

            occlusion (int): Indicating occlusion state with 0 = fully visible, 1 = partly occluded,
                2 = largely occluded, 3 = unknown

            alpha (float): Observation angle of object, ranging [-pi..pi]

            bounding_box (list of floats): 2D bounding box of object in the image (0-based index)
                contains left, top, right, bottom pixel coordinates

            dimensions (list of floats): 3D object dimensions: height, width, length (in meters)

            location (list of floats): 3D object location x,y,z in camera coordinates (in meters)

            rotation (float): Rotation around Y-axis in camera coordinates [-pi..pi]
        """

        def __init__(self, object_idx=None, object_type=None, truncation=None, occlusion=None, alpha=None,
                     bounding_box=None, dimensions=None, location=None, rotation=None):
            self.object_idx = object_idx
            self.object_type = object_type
            self.truncation = truncation
            self.occlusion = occlusion
            self.alpha = alpha
            self.bounding_box = bounding_box
            self.dimensions = dimensions
            self.location = location
            self.rotation = rotation

    class KittiCalibration:
        """Kitti data point calibration data.

        Attributes:
            p0 (ndarray): Camera 1 projection matrix of size (3,4), left gray scale camera
            p1 (ndarray): Camera 2 projection matrix of size (3,4), right gray scale camera
            p2 (ndarray): Camera 3 projection matrix of size (3,4), left color camera
            p3 (ndarray): Camera 4 projection matrix of size (3,4), right color camera
            r0_rect (ndarray): Rectifying rotation matrix of size (4,4)
            tr_vel_to_cam (ndarray): Transformation matrix of size (4,4), velodyne to camera
            tr_imu_to_vel (ndarray): Transformation matrix of size (4,4), imu to velodyne

        Example:
            Project a point from Velodyne coordinates into images for camera i = {0,1,2,3} do:
                x = p_i * r0_rect * tr_vel_to_cam * y
            Project a point from IMU/GPS coordinates into images for camera i = {0,1,2,3} do:
                x = p_i * r0_rect * tr_vel_to_cam * tr_imu_to_vel * y
        """

        def __init__(self, p0=None, p1=None, p2=None, p3=None,
                     r0_rect=None, tr_vel_to_cam=None, tr_imu_to_vel=None):
            self.p0 = p0
            self.p1 = p1
            self.p2 = p2
            self.p3 = p3
            self.r0_rect = r0_rect
            self.tr_vel_to_cam = tr_vel_to_cam
            self.tr_imu_to_vel = tr_imu_to_vel
