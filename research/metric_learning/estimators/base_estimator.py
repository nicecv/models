# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Base estimator defining TCN training, test, and inference."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta
from abc import abstractmethod
import os
import numpy as np
import numpy as np
import data_providers
import preprocessing
from utils import util
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.training import session_run_hook

tf.app.flags.DEFINE_integer(
    'tf_random_seed', 0, 'Random seed.')
FLAGS = tf.app.flags.FLAGS


class InitFromPretrainedCheckpointHook(session_run_hook.SessionRunHook):
  """Hook that can init graph from a pretrained checkpoint."""

  def __init__(self, pretrained_checkpoint_dir):
    """Initializes a `InitFromPretrainedCheckpointHook`.

    Args:
      pretrained_checkpoint_dir: The dir of pretrained checkpoint.

    Raises:
      ValueError: If pretrained_checkpoint_dir is invalid.
    """
    if pretrained_checkpoint_dir is None:
      raise ValueError('pretrained_checkpoint_dir must be specified.')
    self._pretrained_checkpoint_dir = pretrained_checkpoint_dir

  def begin(self):
    checkpoint_reader = tf.contrib.framework.load_checkpoint(
        self._pretrained_checkpoint_dir)
    variable_shape_map = checkpoint_reader.get_variable_to_shape_map()

    exclude_scopes = 'logits/,final_layer/,aux_'
    # Skip restoring global_step as to run fine tuning from step=0.
    exclusions = ['global_step']
    if exclude_scopes:
      exclusions.extend([scope.strip() for scope in exclude_scopes.split(',')])

    variable_to_restore = tf.contrib.framework.get_model_variables()

    # Variable filtering by given exclude_scopes.
    filtered_variables_to_restore = {}
    for v in variable_to_restore:
      excluded = False
      for exclusion in exclusions:
        if v.name.startswith(exclusion):
          excluded = True
          break
      if not excluded:
        var_name = v.name.split(':')[0]
        filtered_variables_to_restore[var_name] = v

    # Final filter by checking shape matching and skipping variables that
    # are not in the checkpoint.
    final_variables_to_restore = {}
    for var_name, var_tensor in filtered_variables_to_restore.iteritems():
      if var_name not in variable_shape_map:
        # Try moving average version of variable.
        var_name = os.path.join(var_name, 'ExponentialMovingAverage')
        if var_name not in variable_shape_map:
          tf.logging.info(
              'Skip init [%s] because it is not in ckpt.', var_name)
          # Skip variables not in the checkpoint.
          continue

      if not var_tensor.get_shape().is_compatible_with(
          variable_shape_map[var_name]):
        # Skip init variable from ckpt if shape dismatch.
        tf.logging.info(
            'Skip init [%s] from [%s] in ckpt because shape dismatch: %s vs %s',
            var_tensor.name, var_name,
            var_tensor.get_shape(), variable_shape_map[var_name])
        continue

      tf.logging.info('Init %s from %s in ckpt' % (var_tensor, var_name))
      final_variables_to_restore[var_name] = var_tensor

    self._init_fn = tf.contrib.framework.assign_from_checkpoint_fn(
        self._pretrained_checkpoint_dir,
        final_variables_to_restore)

  def after_create_session(self, session, coord):
    tf.logging.info('Restoring InceptionV3 weights.')
    self._init_fn(session)
    tf.logging.info('Done restoring InceptionV3 weights.')


class BaseEstimator(object):
  """Abstract TCN base estimator class."""
  __metaclass__ = ABCMeta

  def __init__(self, config, logdir):
    """Constructor.

    Args:
      config: A Luatable-like T object holding training config.
      logdir: String, a directory where checkpoints and summaries are written.
    """
    self._config = config
    self._logdir = logdir

  @abstractmethod
  def construct_input_fn(self, records, is_training):
    """Builds an estimator input_fn.

    The input_fn is used to pass feature and target data to the train,
    evaluate, and predict methods of the Estimator.

    Method to be overridden by implementations.

    Args:
      records: A list of Strings, paths to TFRecords with image data.
      is_training: Boolean, whether or not we're training.

    Returns:
      Function, that has signature of ()->(dict of features, target).
        features is a dict mapping feature names to `Tensors`
        containing the corresponding feature data (typically, just a single
        key/value pair 'raw_data' -> image `Tensor` for TCN.
        labels is a 1-D int32 `Tensor` holding labels.
    """
    pass

  def preprocess_data(self, images, is_training):
    """Preprocesses raw images for either training or inference.

    Args:
      images: A 4-D float32 `Tensor` holding images to preprocess.
      is_training: Boolean, whether or not we're in training.

    Returns:
      data_preprocessed: data after the preprocessor.
    """
    config = self._config
    height = config.data.height
    width = config.data.width
    min_scale = config.data.augmentation.minscale
    max_scale = config.data.augmentation.maxscale
    p_scale_up = config.data.augmentation.proportion_scaled_up
    aug_color = config.data.augmentation.color
    fast_mode = config.data.augmentation.fast_mode
    crop_strategy = config.data.preprocessing.eval_cropping
    preprocessed_images = preprocessing.preprocess_images(
        images, is_training, height, width,
        min_scale, max_scale, p_scale_up,
        aug_color=aug_color, fast_mode=fast_mode,
        crop_strategy=crop_strategy)
    return preprocessed_images

  @abstractmethod
  def forward(self, images, is_training, reuse=False):
    """Defines the forward pass that converts batch images to embeddings.

    Method to be overridden by implementations.

    Args:
      images: A 4-D float32 `Tensor` holding images to be embedded.
      is_training: Boolean, whether or not we're in training mode.
      reuse: Boolean, whether or not to reuse embedder.
    Returns:
      embeddings: A 2-D float32 `Tensor` holding embedded images.
    """
    pass

  @abstractmethod
  def define_loss(self, embeddings, labels, is_training):
    """Defines the loss function on the embedding vectors.

    Method to be overridden by implementations.

    Args:
      embeddings: A 2-D float32 `Tensor` holding embedded images.
      labels: A 1-D int32 `Tensor` holding problem labels.
      is_training: Boolean, whether or not we're in training mode.

    Returns:
      loss: tf.float32 scalar.
    """
    pass

  @abstractmethod
  def define_eval_metric_ops(self):
    """Defines the dictionary of eval metric tensors.

    Method to be overridden by implementations.

    Returns:
      eval_metric_ops:  A dict of name/value pairs specifying the
        metrics that will be calculated when the model runs in EVAL mode.
    """
    pass

  def get_train_op(self, loss):
    """Creates a training op.

    Args:
      loss: A float32 `Tensor` representing the total training loss.
    Returns:
      train_op: A slim.learning.create_train_op train_op.
    Raises:
      ValueError: If specified optimizer isn't supported.
    """
    # Get variables to train (defined in subclass).
    assert self.variables_to_train

    # Define a learning rate schedule.
    decay_steps = self._config.learning.decay_steps
    decay_factor = self._config.learning.decay_factor
    learning_rate = float(self._config.learning.learning_rate)

    # Define a learning rate schedule.
    global_step = slim.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(
        learning_rate,
        global_step,
        decay_steps,
        decay_factor,
        staircase=True)

    # Create an optimizer.
    opt_type = self._config.learning.optimizer
    if opt_type == 'adam':
      opt = tf.train.AdamOptimizer(learning_rate)
    elif opt_type == 'momentum':
      opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
    elif opt_type == 'rmsprop':
      opt = tf.train.RMSPropOptimizer(learning_rate, momentum=0.9,
                                      epsilon=1.0, decay=0.9)
    else:
      raise ValueError('Unsupported optimizer %s' % opt_type)

    # Create a training op.
    # train_op = opt.minimize(loss, var_list=self.variables_to_train)
    # Create a training op.
    train_op = slim.learning.create_train_op(
        loss,
        optimizer=opt,
        variables_to_train=self.variables_to_train,
        update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS))

    return train_op

  def _get_model_fn(self):
    """Defines behavior for training, evaluation, and inference (prediction).

    Returns:
      `model_fn` for `Estimator`.
    """
    # pylint: disable=unused-argument
    def model_fn(features, labels, mode, params):
      """Build the model based on features, labels, and mode.

      Args:
        features: Dict, strings to `Tensor` input data, returned by the
          input_fn.
        labels: The labels Tensor returned by the input_fn.
        mode: A string indicating the mode. This will be either
          tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.PREDICT,
          or tf.estimator.ModeKeys.EVAL.
        params: A dict holding training parameters, passed in during TPU
          training.

      Returns:
        A tf.estimator.EstimatorSpec specifying train/test/inference behavior.
      """
      is_training = mode == tf.estimator.ModeKeys.TRAIN

      # Get preprocessed images from the features dict.
      batch_preprocessed = features['batch_preprocessed']

      # Do a forward pass to embed data.
      batch_encoded = self.forward(batch_preprocessed, is_training)

      # Optionally set the pretrained initialization function.
      initializer_fn = None
      if mode == tf.estimator.ModeKeys.TRAIN:
        initializer_fn = self.pretrained_init_fn

      # If we're training or evaluating, define total loss.
      total_loss = None
      if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
        loss = self.define_loss(batch_encoded, labels, is_training)
        tf.losses.add_loss(loss)
        total_loss = tf.losses.get_total_loss()

      # If we're training, define a train op.
      train_op = None
      if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = self.get_train_op(total_loss)

      # If we're doing inference, set the output to be the embedded images.
      predictions_dict = None
      if mode == tf.estimator.ModeKeys.PREDICT:
        predictions_dict = {'embeddings': batch_encoded}
        # Pass through additional metadata stored in features.
        for k, v in features.iteritems():
          predictions_dict[k] = v

      # If we're evaluating, define some eval metrics.
      eval_metric_ops = None
      if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = self.define_eval_metric_ops()

      # Define training scaffold to load pretrained weights.
      num_checkpoint_to_keep = self._config.logging.checkpoint.num_to_keep
      saver = tf.train.Saver(
          max_to_keep=num_checkpoint_to_keep)

      
      # Build a scaffold to initialize pretrained weights.
      scaffold = tf.train.Scaffold(
          init_fn=initializer_fn,
          saver=saver,
          summary_op=None)
      return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions=predictions_dict,
          loss=total_loss,
          train_op=train_op,
          eval_metric_ops=eval_metric_ops,
          scaffold=scaffold)
    return model_fn

  def train(self):
    """Runs training."""
    # Get a list of training tfrecords.
    config = self._config
    training_dir = config.data.training
    training_records = util.GetFilesRecursively(training_dir)

    # Define batch size.
    self._batch_size = config.data.batch_size

    # Create a subclass-defined training input function.
    train_input_fn = self.construct_input_fn(
        training_records, is_training=True)

    # Create the estimator.
    estimator = self._build_estimator(is_training=True)

    train_hooks = None

    # Run training.
    estimator.train(input_fn=train_input_fn, hooks=train_hooks,
                    steps=config.learning.max_step)

  def _build_estimator(self, is_training):
    """Returns an Estimator object.

    Args:
      is_training: Boolean, whether or not we're in training mode.

    Returns:
      A tf.estimator.Estimator.
    """
    config = self._config
    save_checkpoints_steps = config.logging.checkpoint.save_checkpoints_steps
    keep_checkpoint_max = self._config.logging.checkpoint.num_to_keep
    
    run_config = tf.estimator.RunConfig().replace(
        model_dir=self._logdir,
        save_checkpoints_steps=save_checkpoints_steps,
        keep_checkpoint_max=keep_checkpoint_max,
        tf_random_seed=FLAGS.tf_random_seed)
    return tf.estimator.Estimator(
        model_fn=self._get_model_fn(),
        config=run_config)

  def inference(
      self, inference_input, checkpoint_path, batch_size=None, **kwargs):
    return self._str_inference(inference_input, checkpoint_path)

  def _str_inference(self, str_images, checkpoint_path):
    """Mode 3: Call this repeatedly to do inference over raw images.

    This mode is for when we we want to do real-time inference over
    some stream of images (represented as numpy arrays).

    Args:
      str_images: A array holding raw images to embed.
      checkpoint_path: String, path to a specific checkpoint to restore.
    Returns:
      (embeddings, raw_image_strings):
        embeddings is a 2-D float32 numpy array holding
        [inferred batch_size, embedding_size] image embeddings.
        raw_image_strings is a 1-D string numpy array holding
        [inferred batch_size] jpeg-encoded image strings.
    """
    
    # If this is the first pass, set up inference graph.
    if not hasattr(self, '_str_inf_tensor_dict'):
      self._setup_str_inference(str_images, checkpoint_path)

    # Convert np_images to embeddings.
    np_tensor_dict = self._sess.run(self._str_inf_tensor_dict, feed_dict={
        self._image_str_placeholder: str_images
    })
    return np_tensor_dict['embeddings'], np_tensor_dict['raw_image_strings']

  def _setup_str_inference(self, str_images, checkpoint_path):
    """Sets up and restores inference graph, creates and caches a Session."""
    tf.logging.info('Restoring model weights.')

    # Define inference over an image placeholder.
    image_str_placeholder = tf.placeholder(dtype=tf.string, shape=(None,))

    def preprocess_func(img_string):
      image_data = preprocessing.decode_image(img_string)
      config = self._config
      height = config.data.height
      width = config.data.width
      crop_strategy = config.data.preprocessing.eval_cropping
      preprocessed_image = preprocessing.preprocess_test_image(image_data, height, width, crop_strategy)
      return preprocessed_image
    # image_data = preprocessing.decode_images(image_str_placeholder)
    # Preprocess batch.
    preprocessed = tf.map_fn(preprocess_func, image_str_placeholder, dtype=tf.float32)
    # Unscale and jpeg encode preprocessed images for display purposes.
    im_strings = preprocessing.unscale_jpeg_encode(preprocessed)

    # Do forward pass to get embeddings.
    embeddings = self.forward(preprocessed, is_training=False)

    # Create a saver to restore model variables.
    tf.train.get_or_create_global_step()
    saver = tf.train.Saver(tf.all_variables())

    self._image_str_placeholder = image_str_placeholder
    self._batch_encoded = embeddings

    self._str_inf_tensor_dict = {
        'embeddings': embeddings,
        'raw_image_strings': image_str_placeholder,
    }

    # Create a session and restore model variables.
    self._sess = tf.Session()
    saver.restore(self._sess, checkpoint_path)
