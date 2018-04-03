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

"""Defines data providers used in training and evaluating TCNs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import random
import numpy as np
import preprocessing
import tensorflow as tf

def parse_exmaple(serialized_example, preprocess_fn, is_training, config):
  features = {
      'image0': tf.FixedLenFeature((), tf.string, default_value=''),
      'image1': tf.FixedLenFeature((), tf.string, default_value=''),
      'format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
      'label': tf.FixedLenFeature((), tf.int64),
  }
  parsed_features = tf.parse_single_example(serialized_example, features)
  def preprocess_func(img_string):
    image = preprocessing.decode_image(img_string)
    height = config.data.height
    width = config.data.width
    min_scale = config.data.augmentation.minscale
    max_scale = config.data.augmentation.maxscale
    p_scale_up = config.data.augmentation.proportion_scaled_up
    aug_color = config.data.augmentation.color
    fast_mode = config.data.augmentation.fast_mode
    preprocessed_image = preprocessing.preprocess_training_image(
        image, height, width, min_scale, max_scale, p_scale_up,
        aug_color=aug_color, fast_mode=fast_mode)
    return preprocessed_image

  return preprocess_func(parsed_features['image0']), preprocess_func(parsed_features['image1']), parsed_features['label']

def image_pairs_provider(file_list,
                         config,
                         preprocess_fn,
                         is_training,
                         batch_size,
                         num_parallel_calls):
  
  def _parse_example(serialized_example):
    return parse_exmaple(serialized_example, preprocess_fn, is_training, config)
  
  dataset = tf.data.TFRecordDataset(file_list)
  
  dataset = dataset.shuffle(buffer_size=1024)

  dataset = dataset.map(
      _parse_example, num_parallel_calls=num_parallel_calls).prefetch(batch_size)

  dataset = dataset.repeat()

  dataset = dataset.batch(batch_size//2)

  iterator = dataset.make_one_shot_iterator()
  image0, image1, label = iterator.get_next()
  batch_images = tf.concat([image0, image1], axis=0)
  return batch_images, label
