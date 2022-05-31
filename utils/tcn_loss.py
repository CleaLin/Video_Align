# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

r"""TCN loss for unsupervised training."""

import tensorflow.compat.v2 as tf
from tcc_config import CONFIG


def _npairs_loss(labels, embeddings_anchor, embeddings_positive, reg_lambda):
  """Returns n-pairs metric loss."""
  reg_anchor = tf.reduce_mean(tf.reduce_sum(tf.square(embeddings_anchor), 1))
  reg_positive = tf.reduce_mean(tf.reduce_sum(
      tf.square(embeddings_positive), 1))
  l2loss = 0.25 * reg_lambda * (reg_anchor + reg_positive)

  # Get per pair similarities.
  similarity_matrix = tf.matmul(
      embeddings_anchor, embeddings_positive, transpose_a=False,
      transpose_b=True)

  # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
  lshape = tf.shape(labels)
  assert lshape.shape == 1
  labels = tf.reshape(labels, [lshape[0], 1])

  labels_remapped = tf.cast(
      tf.equal(labels, tf.transpose(labels)), tf.float32)
  labels_remapped /= tf.reduce_sum(labels_remapped, 1, keepdims=True)

  # Add the softmax loss.
  xent_loss = tf.nn.softmax_cross_entropy_with_logits(
      logits=similarity_matrix, labels=labels_remapped)
  xent_loss = tf.reduce_mean(xent_loss)

  return l2loss + xent_loss


def single_sequence_loss(embs, num_steps):
  """Returns n-pairs loss for a single sequence."""

  labels = tf.range(num_steps)
  labels = tf.stop_gradient(labels)
  embeddings_anchor = embs[0::2]
  embeddings_positive = embs[1::2]
  loss = _npairs_loss(labels, embeddings_anchor, embeddings_positive,
                      reg_lambda=CONFIG.REG_LAMBDA)
  return loss

def compute_tcn_loss(embs, training):
  #print('[CLEA] embs shape = ', embs.get_shape()) # (BATCH_SIZE, NUM_STEPS, 128)
  if training:
    num_steps = CONFIG.NUM_STEPS
    batch_size = CONFIG.BATCH_SIZE
  else:
    num_steps = CONFIG.NUM_STEPS
    batch_size = CONFIG.BATCH_SIZE
  losses = []
  for i in range(batch_size):
    # Number of steps is halved due to sampling of positives and anchors.
    losses.append(single_sequence_loss(embs[i], num_steps/2))
  loss = tf.reduce_mean(tf.stack(losses))
  return loss