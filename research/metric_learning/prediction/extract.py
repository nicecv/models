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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import scipy.spatial
import os
import random
import cv2
import math
import numpy as np
from scipy.misc import imresize
from scipy.misc import imsave
from estimators.get_estimator import get_estimator
from utils import util
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
tf.logging.set_verbosity(tf.logging.INFO)

tf.flags.DEFINE_string(
    'config_paths', '',
    """
    Path to a YAML configuration files defining FLAG values. Multiple files
    can be separated by the `#` symbol. Files are merged recursively. Setting
    a key in these files is equivalent to setting the FLAG value with
    the same name.
    """)
tf.flags.DEFINE_string(
    'model_params', '{}', 'YAML configuration string for the model parameters.')
tf.app.flags.DEFINE_string(
    'checkpoint_iter', '', 'Evaluate this specific checkpoint.')
tf.app.flags.DEFINE_string(
    'checkpointdir', '/raid/home/fengfangxiang/log/metric_learning', 'Path to model checkpoints.')
tf.app.flags.DEFINE_string(
    'embedding_file', '/raid/data/nice/metric_data/checked_logs/0004/info/embeddings.txt', 'Path to write embedding info to.')
tf.app.flags.DEFINE_string(
    'query_image_dir', '/raid/data/nice/metric_data/crop_images/query/huarache', 'path to query images')
tf.app.flags.DEFINE_string(
    'candidate_image_dir', '/raid/data/nice/metric_data/crop_images/candidate/huarache', 'path to query images')
FLAGS = tf.app.flags.FLAGS

def get_str_images(dir_path):
  str_images = []
  names = []
  for filename in os.listdir(dir_path):
    path = os.path.join(dir_path, filename)
    image = open(path, 'r').read()
    names.append(filename)
    str_images.append(image)
  return str_images, names

def main(_):
  """Runs main labeled eval loop."""
  # Parse config dict from yaml config files / command line flags.
  config = util.ParseConfigsToLuaTable(FLAGS.config_paths, FLAGS.model_params)

  # Choose an estimator based on training strategy.
  checkpointdir = FLAGS.checkpointdir
  checkpoint_path = os.path.join(
      '%s/model.ckpt-%s' % (checkpointdir, FLAGS.checkpoint_iter))
  estimator = get_estimator(config, checkpointdir)
  
  query_str_images, query_names = get_str_images(FLAGS.query_image_dir)
  cand_str_images, cand_names = get_str_images(FLAGS.candidate_image_dir)
  query_labels = [1]*len(query_str_images)
  cand_labels = [2]*len(cand_str_images)
  
  str_images = query_str_images + cand_str_images
  names = query_names + cand_names
  labels = query_labels + cand_labels
  
  bsize = 16
  nbatch = int(math.ceil(len(labels) / float(bsize)))
  res = np.zeros((len(labels), config.embedding_size))
  for n in xrange(nbatch):
    s = n*bsize
    e = min((n+1)*bsize,len(labels))
    (embeddings, _) = estimator.inference(str_images[s:e], checkpoint_path)
    res[s:e] = embeddings
    print (n)
  with open(FLAGS.embedding_file, 'w') as fw:
    for i in xrange(len(names)):
      fw.write(names[i]+'\t'+str(labels[i])+'\t'+','.join([str(v) for v in res[i,:]])+'\n')

if __name__ == '__main__':
  tf.app.run(main)
