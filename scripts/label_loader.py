import xml.etree.ElementTree as ET

from classes.IndividualObject import IndividualObject


def load_labels(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    items = root.findall('./tracklets/item')

    for item in items:

        object_type = item.find('./objectType')

        height = item.find('./h')
        width = item.find('./w')
        length = item.find('./l')

        first_frame = item.find('./first_frame')
        poses = item.find('./poses')
        finished = item.find('./finished')

        individual_object = IndividualObject(object_type, height, width, length,
                                             first_frame, finished)

        for i in poses.findall('./item'):
            tx = i.find('./tx')
            ty = i.find('./ty')
            tz = i.find('./tz')

            rx = i.find('./rx')
            ry = i.find('./ry')
            rz = i.find('./rz')

            state = i.find('./state')

            occlusion = i.find('./occlusion')
            occlusion_kf = i.find('./occlusion_kf')

            truncation = i.find('./truncation')

            amt_occlusion = i.find('./amt_occlusion')
            amt_occlusion_kf = i.find('./amt_occlusion_kf')

            amt_border_l = i.find('./amt_border_l')
            amt_border_r = i.find('./amt_border_r')
            amt_border_kf = i.find('./amt_border_kf')

            individual_object.add_pose(tx, ty, tz, rx, ry, rz, state,
                                       occlusion, occlusion_kf, truncation,
                                       amt_occlusion, amt_occlusion_kf,
                                       amt_border_l, amt_border_r, amt_border_kf)


def main():
    load_labels('E:\\Documents\\KITTI\Raw\\2011_09_26\part0\\2011_09_26_drive_0001_sync\\tracklet_labels.xml')


if __name__ == '__main__':
    main()
