import os
import urllib.request
import tarfile
import logging
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
LOGGER = logging.getLogger(__name__)
RESNET_SCOPE_NAME = 'resnet_v1_50'
RESNET_V1_50 = 'http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz'
def restore_origin_resnet(session, from_dir='/tmp', exclude_vars=None):
    """Restore parameters from existing checkpoint.
    Args:
        session (tf.Session): Tensorflow session, used to perform the restore operation
        from_dir (str): Full path to the directory where we can find the checkpoint
        exclude_vars (list of str, optional): Variables to exclude from checkpoint restoration
    """
    checkpoint = _get_checkpoint(RESNET_V1_50, ckpt_dir=from_dir, ckpt_name='resnet_v1_50.ckpt')
    variables_to_restore = slim.get_variables_to_restore(include=[RESNET_SCOPE_NAME], exclude=exclude_vars)
    saver = tf.train.Saver(variables_to_restore)
    saver.restore(session, checkpoint)
    LOGGER.info('Restore original resnet checkpoint done!')
def _get_checkpoint(url, ckpt_dir='/tmp', ckpt_name='resnet_v1_50.ckpt'):
    """Retrieve pre-trained network, if no found locally, get it from internet.
    Args:
        ckpt_dir (str): the directory to hold checkpoint file
        ckpt_name (str): the checkpoint file name
        url (str): url from which to download the pre-trained network
    Returns:
        the path to the downloaded network's checkpoint file
    """
    target_ckpt_file = os.path.realpath(os.path.join(ckpt_dir, ckpt_name))
    if os.path.isfile(target_ckpt_file):
        LOGGER.info('The network checkpoint is already exist: %s', target_ckpt_file)
    else:
        LOGGER.info('Downloading pre-trained network from url: %s ...', url)
        tar_file, _ = urllib.request.urlretrieve(url)
        with tarfile.open(tar_file, mode='r|gz') as tar:
            tar.extractall(path=ckpt_dir)
        assert os.path.isfile(target_ckpt_file)
        LOGGER.info('Successfully get network checkpoint: %s', target_ckpt_file)
    return target_ckpt_file
def forward(images, num_classes, upsampling_factor=8):
    """Perform forward operation on thesis net.
    Args:
        images (TF placeholder): Network input placeholder
            [batches, height, width, channels]
        num_classes: Number of class for classification
        upsampling_factor (int): Upsample factor
    Returns:
        class_head (Tensor): Model output for logits, with shape
            [batches, num_classes, height, width]
        bbox_head (Tensor): Model output for Bounding box, with shape
            [batches, 4, height, width]
        depth_head (Tensor): Model output for depth, with shape
            [batches, 1, height, width]
    """
    resnet_features = _get_resnet_features(images)
    class_head = _create_task_net(resnet_features, num_classes,
                                  upsampling_factor=upsampling_factor)
    class_head = tf.nn.softmax(class_head, dim=1)
    bbox_head = _create_task_net(resnet_features, 4,
                                 upsampling_factor=upsampling_factor)
    depth_head = _create_task_net(resnet_features, 1,
                                  upsampling_factor=upsampling_factor)
    return class_head, bbox_head, depth_head
def _get_resnet_features(inputs):
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        resnet_v1.resnet_v1_50(inputs, num_classes=None, is_training=True)
    return tf.get_default_graph().get_tensor_by_name('resnet_v1_50/block4/unit_3/bottleneck_v1/Relu:0')
def _create_task_net(inputs, output_features, upsampling_factor):
    """Create thesis task net branch.
    Args:
        inputs (Tensor): A Tensor of rank N+2 of shape
            [batch_size] + input_spatial_shape + [in_channels]
        output_features (int): Number of output features
        upsampling_factor (int): Upsample factor
    Returns:
        Tensor with shape [batches, output_features, 88, 304]
    """
    inputs_shape = inputs.get_shape().as_list()
    batches = inputs_shape[0]
    in_channels = inputs_shape[-1]
    net = slim.conv2d(inputs, in_channels * 2, [1, 1], stride=1)
    net = slim.batch_norm(net)
    net = tf.nn.relu(net)
    net = slim.conv2d(net, output_features * upsampling_factor ** 2, [1, 1], stride=1)
    net = tf.reshape(net, [batches, output_features, 88, 304])
    return net