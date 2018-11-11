"""Tensorflow train script."""

import os
import time
import logging
import tensorflow as tf

from common.datareaders import KittiDataReader
from common.datasamplers import RandomDataSampler
from common.dataloader import DataLoader
from fw_tensorflow import model
from fw_tensorflow.dataset import ThesisNetKittiDataset

LOGGER = logging.getLogger(__name__)
LOG_FORMAT = '%(asctime)-15s %(levelname)s %(name)s - %(message)s'

PROJECT_PATH = os.getcwd()
EXPERIMENT_PATH = os.path.join(PROJECT_PATH, time.strftime("%c").replace(':', '_'))


def get_kitti_loader(dataset_path, num_classes=5, stride=4, batch_size=1, epochs=1, drop_last=True,
                     num_workers=1, data_sampler=None, use_multiprocessing=True):
    """Get Kitti dataset loader."""

    "Create a KittiDataReader instance to read image, label and calibration files"
    reader = KittiDataReader(dataset_path=dataset_path,
                             read_color_left=True,
                             read_color_right=False,
                             read_calib=False,
                             read_labels=True)

    dataset = ThesisNetKittiDataset(data_readers=[reader],
                                    num_classes=num_classes,
                                    stride=stride)

    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        epochs=epochs,
                        drop_last=drop_last,
                        data_sampler=data_sampler,
                        num_workers=num_workers,
                        use_multiprocessing=use_multiprocessing)
    return loader


def get_model_placeholders(loader):
    """Creates model placeholders.
    Args:
        loader (DataLoader): The data loader
    Returns:
        images_ph (tf.placeholder): input images
        classes_ph (tf.placeholder): class target masks
        bboxes_ph (tf.placeholder): bbox target masks
        depths_ph (tf.placeholder): depth target masks
        suppress_ph (tf.placeholder): suppress masks
    """

    batch_size = loader.batch_size
    num_classes = loader.dataset.target_generator.num_classes
    num_coords = loader.dataset.target_generator.num_coords
    num_depth = loader.dataset.target_generator.num_depth
    target_height = loader.dataset.target_generator.target_height
    target_width = loader.dataset.target_generator.target_width
    image_height = loader.dataset.kitti_reader.image_height
    image_width = loader.dataset.kitti_reader.image_width
    image_channels = loader.dataset.kitti_reader.image_channels
    images_ph = tf.placeholder(dtype=tf.float32, shape=[batch_size, image_height, image_width, image_channels])
    classes_ph = tf.placeholder(dtype=tf.float32, shape=[batch_size, num_classes, target_height, target_width])
    bboxes_ph = tf.placeholder(dtype=tf.float32, shape=[batch_size, num_coords, target_height, target_width])
    depths_ph = tf.placeholder(dtype=tf.float32, shape=[batch_size, num_depth, target_height, target_width])
    suppress_ph = tf.placeholder(dtype=tf.float32, shape=[batch_size, target_height, target_width])
    return images_ph, classes_ph, bboxes_ph, depths_ph, suppress_ph


def get_loss(class_targets, class_predictions, bbox_targets, bbox_predictions, depth_targets, depth_predictions,
             class_weight=1, bbox_weight=1, depth_weight=1):
    """Defines the loss function.
    Args:
        class_targets (tf.placeholder): classification target mask
        class_predictions (tf.tensor): classification prediction
        bbox_targets (tf.placeholder): bounding box target mask
        bbox_predictions (tf.tensor): bounding box prediction
        depth_targets (tf.placeholder): depth target mask
        depth_predictions (tf.tensor): depth prediction
        class_weight (int, optional): loss weigh for classification task
        bbox_weight (int, optional): loss weight for bounding box task
        depth_weight (int, optional): loss weight for depth task
    Returns:
        A tf.tensor loss
    """
    class_loss = tf.losses.mean_squared_error(labels=class_targets, predictions=class_predictions, weights=class_weight)
    bbox_loss = tf.losses.mean_squared_error(labels=bbox_targets, predictions=bbox_predictions, weights=bbox_weight)
    depth_loss = tf.losses.mean_squared_error(labels=depth_targets, predictions=depth_predictions, weights=depth_weight)
    return class_loss + bbox_loss + depth_loss


def main():
    """Train function."""
    batch_size = 1
    learning_rate = 0.0001
    loader = get_kitti_loader(dataset_path='E:\Documents\KITTI\Images',
                              batch_size=batch_size,
                              data_sampler=RandomDataSampler)
    with tf.Session() as sess:
        try:
            # Start data loader
            loader.request_start()
            # Step variable
            global_step = tf.Variable(0, name='global_step', trainable=False)
            # Get input/target placeholder
            images_ph, class_targets_ph, bbox_targets_ph, depth_targets_ph, suppress_ph = get_model_placeholders(loader)
            # Get model predictions
            class_predictions, bbox_predictions, depth_predictions = model.forward(images_ph, loader.dataset.target_generator.num_classes)
            # Get loss op
            loss = get_loss(class_targets_ph, class_predictions, bbox_targets_ph, bbox_predictions,
                            depth_targets_ph, depth_predictions)
            # Define optimizer
            with tf.variable_scope('adamoptim'):
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                train = optimizer.minimize(loss=loss, global_step=global_step)
            # Define train op
            sess.run(tf.global_variables_initializer())
            model.restore_origin_resnet(session=sess, exclude_vars=['adamoptim'])
            # Run training loop
            while not loader.should_stop():
                images, class_targets, bbox_targets, depth_targets, suppress = loader.next()
                _, step, loss_ = sess.run([train, global_step, loss], feed_dict={images_ph: images,
                                                                                 class_targets_ph: class_targets,
                                                                                 bbox_targets_ph: bbox_targets,
                                                                                 depth_targets_ph: depth_targets,
                                                                                 suppress_ph: suppress})
                LOGGER.info('step: %s, loss: %s', step, loss_)
        finally:
            loader.request_stop()


def setup_logging(log_path, debug):
    """Setup logging for the script.
    Args:
        log_path (str): full path to the desired log file
        debug (bool): log in verbose mode or not
    """

    "Create the logger and set the level of information being saved"
    lvl = logging.DEBUG if debug else logging.INFO
    root_logger = logging.getLogger()
    root_logger.setLevel(lvl)
    fmt = logging.Formatter(fmt=LOG_FORMAT)

    "Create stream handler for the logger"
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)

    "Create file handler which writes formatted logging records to logging files and set the format"
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)

    "Add stream and file handler to the logger"
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)


if __name__ == '__main__':
    setup_logging(log_path=EXPERIMENT_PATH, debug=False)
    main()
