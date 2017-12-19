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
from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim
    
def prepare_resized_image_to_newsize(image, new_height, new_width):
  original_shape = tf.shape(image)
  if original_shape[0] == new_height and original_shape[1] == new_width:
    return image
  image = tf.expand_dims(image, 0)
  resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                         align_corners=False)
  resized_image = tf.squeeze(resized_image)
  resized_image.set_shape([new_height, new_width, 3])
  return resized_image
  
class APPBaseModel(object):
  def __init__(self, 
      model_name,
      checkpoint_path, 
      num_classes,
      preprocessing_name=None,
      labels_offset=0, 
      prepare_image_size=0,
      outputs=['predictions'],
      label_file=None):
    
    if label_file:
      self._label2text = [line.strip() for line in open(label_file, 'r')]
    
    network_fn = nets_factory.get_network_fn(
        model_name,
        num_classes=(num_classes - labels_offset),
        is_training=False)
    if preprocessing_name is None:
      preprocessing_name = model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)
    image_size = network_fn.default_image_size
    
    self._placeholder = tf.placeholder(name='input', dtype=tf.string, shape=(None,))
    
    def preprocess_func(img_string):
      img = tf.image.decode_jpeg(img_string, channels=3)
      if prepare_image_size:
        img = prepare_resized_image_to_newsize(img, prepare_image_size, prepare_image_size)
      img = image_preprocessing_fn(img, image_size, image_size)
      return img
    
    def crop_image_func(img_string):
      img = tf.image.decode_jpeg(img_string, channels=3)
      if prepare_image_size:
        img = prepare_resized_image_to_newsize(img, prepare_image_size, prepare_image_size)
      image_height = tf.shape(img)[0]
      image_width = tf.shape(img)[1]
  
      offset_height = tf.to_int32((image_height - image_size) / 2)
      offset_width = tf.to_int32((image_width - image_size) / 2)
      return tf.image.crop_to_bounding_box(img, offset_height, offset_width, image_size, image_size)
    
    self._crop_images = tf.map_fn(crop_image_func, self._placeholder, dtype=tf.float32)
    
    processed_images = tf.map_fn(preprocess_func, self._placeholder, dtype=tf.float32)
    _, end_points = network_fn(processed_images)
    
    end_points['processed_images'] = processed_images
    self._layers = end_points
    
    self._features = {}
    for output in outputs:
      if output not in self._layers.keys():
        raise KeyError('output name must in ['+','.join(self._layers.keys())+']')
      self._features[output] = end_points[output]
      
    init_fn = slim.assign_from_checkpoint_fn(
        checkpoint_path,
        slim.get_model_variables(model_name))
    
    self._sess = tf.Session()
    init_fn(self._sess)
  
  def get_crop_images(self, images):
    return self._sess.run(self._crop_images, feed_dict={self._placeholder:images})
    
  def predict(self, images):
    features = self._sess.run(self._features, feed_dict={self._placeholder:images})
    return features
  
  def get_weights(self, var_name):
    exe_var = [var for var in self.get_all_variables() if var.name==var_name]
    if len(exe_var) > 0:
      return self._sess.run(exe_var[0])
    return None
    
  def get_all_variables(self):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
  
  def get_all_layers(self):
    return self._layers