class Pose:
    def __init__(self, tx, ty, tz, rx, ry, rz, state, occlusion, occlusion_kf,
                 truncation, amt_occlusion, amt_occlusion_kf, amt_border_l,
                 amt_border_r, amt_border_kf):

        self.tx = tx
        self.ty = ty
        self.tz = tz

        self.rx = rx
        self.ry = ry
        self.rz = rz

        self.state = state

        self.occlusion = occlusion
        self.occlusion_kf = occlusion_kf

        self.truncation = truncation

        self.amt_occlusion = amt_occlusion
        self.amt_occlusion_kf = amt_occlusion_kf

        self.amt_border_l = amt_border_l
        self.amt_border_r = amt_border_r
        self.amt_border_kf = amt_border_kf

    def __str__(self):
        print('tx:\t{}'.format(self.tx))
        print('ty:\t{}'.format(self.ty))
        print('tz:\t{}\r'.format(self.tz))

        print('rx:\t{}'.format(self.rx))
        print('ry:\t{}'.format(self.ry))
        print('rz:\t{}\r'.format(self.rz))

        print('State:\t\t\t{}'.format(self.state))

        print('Occlusion:\t\t{}'.format(self.occlusion))
        print('Occlusion_kf:\t{}\r'.format(self.occlusion_kf))

        print('Truncation:\t\t\t{}\r'.format(self.truncation))

        print('Amt_occlusion:\t\t{}'.format(self.amt_occlusion))
        print('Amt_occlusion_kf:\t{}\r'.format(self.amt_occlusion_kf))

        print('Amt_border_l:\t{}'.format(self.amt_border_l))
        print('Amt_border_r:\t{}'.format(self.amt_border_r))
        print('Amt_border_kf:\t{}'.format(self.amt_border_kf))
