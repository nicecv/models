# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to save.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')
    
tf.app.flags.DEFINE_boolean(
    'is_training', False,
    'Whether to save out a training-focused version of the model.')

tf.app.flags.DEFINE_integer(
    'image_size', None,
    'The image size to use, otherwise use the model default_image_size.')

tf.app.flags.DEFINE_integer(
    'prepare_image_size', None,
    'The image size to use.')


tf.app.flags.DEFINE_string('dataset_name', 'imagenet',
                           'The name of the dataset to use with the model.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'input_name', 'input', 'The name of input tensor')
tf.app.flags.DEFINE_string(
    'output_name', 'output', 'The name of output tensor')

tf.app.flags.DEFINE_string(
    'output_file', '', 'Where to save the resulting file to.')

tf.app.flags.DEFINE_string(
    'dataset_dir', '', 'Directory to save intermediate dataset files to')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '', 'Model file')

FLAGS = tf.app.flags.FLAGS

def prepare_resized_image_to_newsize(image, new_height, new_width):
    original_shape = tf.shape(image)
    if original_shape[0] == new_height and original_shape[1] == new_width:
        return image
    image = tf.expand_dims(image, 0)
    resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                           align_corners=False)
    resized_image = tf.squeeze(resized_image)
    resized_image.set_shape([None, None, 3])
    return resized_image

def main(_):
  if not FLAGS.output_file:
    raise ValueError('You must supply the path to save to with --output_file')
  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default() as graph:
    dataset = dataset_factory.get_dataset(FLAGS.dataset_name, 'train',
                                          FLAGS.dataset_dir)
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        is_training=FLAGS.is_training)
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=FLAGS.is_training)
    image_size = FLAGS.image_size or network_fn.default_image_size
    
    placeholder = tf.placeholder(name=FLAGS.input_name, dtype=tf.string, shape=(None,))
    def preprocess_func(img_string):
      img = tf.image.decode_jpeg(img_string, channels=3)
      if FLAGS.prepare_image_size:
        # first resize the image to prepare_image_size
        prepare_image_size = FLAGS.prepare_image_size
        img = prepare_resized_image_to_newsize(img, prepare_image_size, prepare_image_size) 
      img = image_preprocessing_fn(img, image_size, image_size)
      return img
    
    processed_images = tf.map_fn(preprocess_func, placeholder, dtype=tf.float32)
    
    logits, _ = network_fn(processed_images)
    
    #probabilities = tf.nn.softmax(logits, name='prob')
    
    preds_values, preds_indices = tf.nn.top_k(logits, k=10, name=FLAGS.output_name)
    
    init_fn = slim.assign_from_checkpoint_fn(
                FLAGS.checkpoint_path,
                slim.get_model_variables(FLAGS.model_name))
    with tf.Session() as sess:
      init_fn(sess)
      output_graph_def = graph_util.convert_variables_to_constants(
          sess, graph.as_graph_def(), [FLAGS.output_name])
      with tf.gfile.FastGFile(FLAGS.output_file, 'wb') as f:
          f.write(output_graph_def.SerializeToString())
    

if __name__ == '__main__':
  tf.app.run()
