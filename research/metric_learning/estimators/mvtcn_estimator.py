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

"""MVTCN trainer implementations with various metric learning losses."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import data_providers
import model as model_module
from estimators import base_estimator
import tensorflow as tf


class MVTCNEstimator(base_estimator.BaseEstimator):
  """Multi-view TCN base class."""

  def __init__(self, config, logdir):
    super(MVTCNEstimator, self).__init__(config, logdir)

  def _pairs_provider(self, records, is_training):
    config = self._config
    num_parallel_calls = config.data.num_parallel_calls
    return functools.partial(
        data_providers.image_pairs_provider,
        file_list=records,
        config=config,
        preprocess_fn=self.preprocess_data,
        is_training=is_training,
        batch_size=self._batch_size,
        num_parallel_calls=num_parallel_calls)

  def forward(self, images_concat, is_training, reuse=False):
    """See base class."""
    embedder_strategy = self._config.embedder_strategy
    loss_strategy = self._config.loss_strategy
    l2_normalize_embedding = self._config[loss_strategy].embedding_l2
    embedder = model_module.get_embedder(
        embedder_strategy,
        self._config,
        images_concat,
        is_training=is_training,
        l2_normalize_embedding=l2_normalize_embedding, reuse=reuse)
    embeddings_concat = embedder.construct_embedding()
    variables_to_train = embedder.get_trainable_variables()
    self.variables_to_train = variables_to_train
    self.pretrained_init_fn = embedder.init_fn
    return embeddings_concat

  def _collect_image_summaries(self, images_concat):
    image_summaries = self._config.logging.summary.image_summaries
    if image_summaries:
      tf.summary.image('training/images_preprocessed_concat', images_concat)


class MVTCNTripletEstimator(MVTCNEstimator):
  """Multi-View TCN with semihard triplet loss."""

  def __init__(self, config, logdir):
    super(MVTCNTripletEstimator, self).__init__(config, logdir)

  def construct_input_fn(self, records, is_training):
    """See base class."""
    def input_fn(params):
      """Provides input to MVTCN models."""
      batch_size = self._batch_size
      (batch_images, labels) = self._pairs_provider(
           records, is_training)(batch_size=batch_size)
      if is_training:
        self._collect_image_summaries(batch_images)
      features = {'batch_preprocessed': batch_images}
      return (features, labels)
    return input_fn

  def define_loss(self, embeddings, labels, is_training):
    """See base class."""
    margin = self._config.triplet_semihard.margin
    loss = tf.contrib.losses.metric_learning.triplet_semihard_loss(
        labels=labels, embeddings=embeddings, margin=margin)
    self._loss = loss
    if is_training:
      tf.summary.scalar('training/triplet_semihard', loss)
    return loss

  def define_eval_metric_ops(self):
    """See base class."""
    return {'validation/triplet_semihard': tf.metrics.mean(self._loss)}


class MVTCNNpairsEstimator(MVTCNEstimator):
  """Multi-View TCN with npairs loss."""

  def __init__(self, config, logdir):
    super(MVTCNNpairsEstimator, self).__init__(config, logdir)

  def construct_input_fn(self, records, is_training):
    """See base class."""
    def input_fn(params):
      """Provides input to MVTCN models."""
      batch_size = self._batch_size
      (batch_images, npairs_labels) = self._pairs_provider(
           records, is_training)(batch_size=batch_size)
      if is_training:
        self._collect_image_summaries(batch_images)
      features = {'batch_preprocessed': batch_images}
      return (features, npairs_labels)
    return input_fn

  def define_loss(self, embeddings, labels, is_training):
    """See base class."""
    embeddings_anchor, embeddings_positive = tf.split(embeddings, 2, axis=0)
    loss = tf.contrib.losses.metric_learning.npairs_loss(
        labels=labels, embeddings_anchor=embeddings_anchor,
        embeddings_positive=embeddings_positive)
    self._loss = loss
    if is_training:
      tf.summary.scalar('training/npairs', loss)
    return loss

  def define_eval_metric_ops(self):
    """See base class."""
    return {'validation/npairs': tf.metrics.mean(self._loss)}
