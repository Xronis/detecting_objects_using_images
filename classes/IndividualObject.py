from .Pose import Pose


class IndividualObject:
    def __init__(self, obj_type, height, width, length, first_frame, finished):
        self.obj_type = obj_type
        self.height = height
        self.width = width
        self.length = length
        self.first_frame = first_frame
        self.finished = finished
        self.poses = []

    def add_pose(self, tx, ty, tz, rx, ry, rz, state, occlusion, occlusion_kf, truncation,
                 amt_occlusion, amt_occlusion_kf, amt_border_l, amt_border_r, amt_border_kf, frame):

        self.poses.append(Pose(tx, ty, tz, rx, ry, rz, state, occlusion, occlusion_kf, truncation,
                               amt_occlusion, amt_occlusion_kf, amt_border_l, amt_border_r, amt_border_kf, frame))

    def __str__(self):
        print('Object Type:\t{}'.format(self.obj_type))
        print('Height:\t\t\t{}'.format(self.height))
        print('Width:\t\t\t{}'.format(self.width))
        print('Length:\t\t\t{}'.format(self.length))
        print('First Frame:\t{}'.format(self.first_frame))
        print('Finished:\t\t{}\n'.format(self.finished))

        for pose in self.poses:
            print('------------------- Pose at frame {} -------------------'.format(pose.frame))
            pose.__str__()
            print('\n')

