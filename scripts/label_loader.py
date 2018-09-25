import xml.etree.ElementTree as ET

from classes.IndividualObject import IndividualObject


def load_labels(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    items = root.findall('./tracklets/item')

    individual_objects = []

    for item in items:

        object_type = item.find('./objectType').text

        height = item.find('./h').text
        width = item.find('./w').text
        length = item.find('./l').text

        first_frame = item.find('./first_frame').text
        poses = item.find('./poses')
        finished = item.find('./finished').text

        individual_object = IndividualObject(object_type, height, width, length,
                                             first_frame, finished)

        item_in_poses = poses.findall('./item')

        for i in range(len(item_in_poses)):
            tx = item_in_poses[i].find('./tx').text
            ty = item_in_poses[i].find('./ty').text
            tz = item_in_poses[i].find('./tz').text

            rx = item_in_poses[i].find('./rx').text
            ry = item_in_poses[i].find('./ry').text
            rz = item_in_poses[i].find('./rz').text

            state = item_in_poses[i].find('./state').text

            occlusion = item_in_poses[i].find('./occlusion').text
            occlusion_kf = item_in_poses[i].find('./occlusion_kf').text

            truncation = item_in_poses[i].find('./truncation').text

            amt_occlusion = item_in_poses[i].find('./amt_occlusion').text
            amt_occlusion_kf = item_in_poses[i].find('./amt_occlusion_kf').text

            amt_border_l = item_in_poses[i].find('./amt_border_l').text
            amt_border_r = item_in_poses[i].find('./amt_border_r').text
            amt_border_kf = item_in_poses[i].find('./amt_border_kf').text

            individual_object.add_pose(tx, ty, tz, rx, ry, rz, state,
                                       occlusion, occlusion_kf, truncation,
                                       amt_occlusion, amt_occlusion_kf,
                                       amt_border_l, amt_border_r, amt_border_kf, i)

        individual_objects.append(individual_object)

    return individual_objects


def main():
    individual_objects = load_labels('C:\\Users\\ppanagiotidis\\Pictures\\Raw\\part0\\2011_09_26\\2011_09_26_drive_0001_sync\\tracklet_labels.xml')

    for obj in individual_objects:
        obj.__str__()


if __name__ == '__main__':
    main()
