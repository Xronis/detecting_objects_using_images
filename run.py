import time
import os

from scripts import image_loader_using_pykitti, label_loader

if __name__ == '__main__':

    start_time = time.time()

    basedir = 'E:\Documents\KITTI\Raw'
    date = '2011_09_26'

    drives = os.listdir(basedir + '\\' + date + '\\')
    drives = [drive for drive in drives if os.path.isdir(basedir + '\\' + date + '\\' + drive)]

    for drive in drives:

        clean_drive = drive.split('_')[-2]
        images = image_loader_using_pykitti.load_images(basedir=basedir, date=date, drive=clean_drive)

        xml_file = basedir + '\\' + date + '\\' + drive + '\\tracklet_labels.xml'
        labels = label_loader.load_labels(xml_file)

        print('Loaded drive ' + clean_drive)

        poses_of_frame = image_loader_using_pykitti.get_poses_of_frame(labels, 1)

        for pose in poses_of_frame:
            pose.__str__()

    print('Execution Time: {} secs'.format(time.time() - start_time))
