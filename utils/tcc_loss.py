import tensorflow as tf

# TCC Loss
def classification_loss(logits, labels, label_smoothing):
  """Loss function based on classifying the correct indices.
  In the paper, this is called Cycle-back Classification.
  Args:
    logits: Tensor, Pre-softmax scores used for classification loss. These are
      similarity scores after cycling back to the starting sequence.
    labels: Tensor, One hot labels containing the ground truth. The index where
      the cycle started is 1.
    label_smoothing: Float, label smoothing factor which can be used to
      determine how hard the alignment should be.
  Returns:
    loss: Tensor, A scalar classification loss calculated using standard softmax
      cross-entropy loss.
  """
  # Just to be safe, we stop gradients from labels as we are generating labels.
  labels = tf.stop_gradient(labels)
  return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(
      y_true=labels, y_pred=logits, from_logits=True,
      label_smoothing=label_smoothing))


def regression_loss(logits, labels, num_steps, steps, seq_lens, loss_type,
                    normalize_indices, variance_lambda, huber_delta):
  """Loss function based on regressing to the correct indices.
  In the paper, this is called Cycle-back Regression. There are 3 variants
  of this loss:
  i) regression_mse: MSE of the predicted indices and ground truth indices.
  ii) regression_mse_var: MSE of the predicted indices that takes into account
  the variance of the similarities. This is important when the rate at which
  sequences go through different phases changes a lot. The variance scaling
  allows dynamic weighting of the MSE loss based on the similarities.
  iii) regression_huber: Huber loss between the predicted indices and ground
  truth indices.
  Args:
    logits: Tensor, Pre-softmax similarity scores after cycling back to the
      starting sequence.
    labels: Tensor, One hot labels containing the ground truth. The index where
      the cycle started is 1.
    num_steps: Integer, Number of steps in the sequence embeddings.
    steps: Tensor, step indices/frame indices of the embeddings of the shape
      [N, T] where N is the batch size, T is the number of the timesteps.
    seq_lens: Tensor, Lengths of the sequences from which the sampling was done.
      This can provide additional temporal information to the alignment loss.
    loss_type: String, This specifies the kind of regression loss function.
      Currently supported loss functions: regression_mse, regression_mse_var,
      regression_huber.
    normalize_indices: Boolean, If True, normalizes indices by sequence lengths.
      Useful for ensuring numerical instabilities don't arise as sequence
      indices can be large numbers.
    variance_lambda: Float, Weight of the variance of the similarity
      predictions while cycling back. If this is high then the low variance
      similarities are preferred by the loss while making this term low results
      in high variance of the similarities (more uniform/random matching).
    huber_delta: float, Huber delta described in tf.keras.losses.huber_loss.
  Returns:
     loss: Tensor, A scalar loss calculated using a variant of regression.
  """
  # Just to be safe, we stop gradients from labels as we are generating labels.
  labels = tf.stop_gradient(labels)
  steps = tf.stop_gradient(steps)

  if normalize_indices:
    float_seq_lens = tf.cast(seq_lens, tf.float32)
    tile_seq_lens = tf.tile(
        tf.expand_dims(float_seq_lens, axis=1), [1, num_steps])
    steps = tf.cast(steps, tf.float32) / tile_seq_lens
  else:
    steps = tf.cast(steps, tf.float32)

  beta = tf.nn.softmax(logits)
  true_time = tf.reduce_sum(steps * labels, axis=1)
  pred_time = tf.reduce_sum(steps * beta, axis=1)

  if loss_type in ['regression_mse', 'regression_mse_var']:
    if 'var' in loss_type:
      # Variance aware regression.
      pred_time_tiled = tf.tile(tf.expand_dims(pred_time, axis=1),
                                [1, num_steps])

      pred_time_variance = tf.reduce_sum(
          tf.square(steps - pred_time_tiled) * beta, axis=1)

      # Using log of variance as it is numerically stabler.
      pred_time_log_var = tf.math.log(pred_time_variance)
      squared_error = tf.square(true_time - pred_time)
      return tf.reduce_mean(tf.math.exp(-pred_time_log_var) * squared_error
                            + variance_lambda * pred_time_log_var)

    else:
      return tf.reduce_mean(
          tf.keras.losses.mean_squared_error(y_true=true_time,
                                             y_pred=pred_time))
  elif loss_type == 'regression_huber':
    return tf.reduce_mean(tf.keras.losses.huber_loss(
        y_true=true_time, y_pred=pred_time,
        delta=huber_delta))
  else:
    raise ValueError('Unsupported regression loss %s. Supported losses are: '
                     'regression_mse, regresstion_mse_var and regression_huber.'
                     % loss_type)
    
    
def pairwise_l2_distance(embs1, embs2):
  """Computes pairwise distances between all rows of embs1 and embs2."""
  norm1 = tf.reduce_sum(tf.square(embs1), 1)
  norm1 = tf.reshape(norm1, [-1, 1])
  norm2 = tf.reduce_sum(tf.square(embs2), 1)
  norm2 = tf.reshape(norm2, [1, -1])

  # Max to ensure matmul doesn't produce anything negative due to floating
  # point approximations.
  dist = tf.maximum(
      norm1 + norm2 - 2.0 * tf.matmul(embs1, embs2, False, True), 0.0)

  return dist


def get_scaled_similarity(embs1, embs2, similarity_type, temperature):
  """Returns similarity between each all rows of embs1 and all rows of embs2.
  The similarity is scaled by the number of channels/embedding size and
  temperature.
  Args:
    embs1: Tensor, Embeddings of the shape [M, D] where M is the number of
      embeddings and D is the embedding size.
    embs2: Tensor, Embeddings of the shape [N, D] where N is the number of
      embeddings and D is the embedding size.
    similarity_type: String, Either one of 'l2' or 'cosine'.
    temperature: Float, Temperature used in scaling logits before softmax.
  Returns:
    similarity: Tensor, [M, N] tensor denoting similarity between embs1 and
      embs2.
  """
  channels = tf.cast(tf.shape(embs1)[1], tf.float32)
  # Go for embs1 to embs2.
  if similarity_type == 'cosine':
    similarity = tf.matmul(embs1, embs2, transpose_b=True)
  elif similarity_type == 'l2':
    similarity = -1.0 * pairwise_l2_distance(embs1, embs2)
  else:
    raise ValueError('similarity_type can either be l2 or cosine.')

  # Scale the distance  by number of channels. This normalization helps with
  # optimization.
  similarity /= channels
  # Scale the distance by a temperature that helps with how soft/hard the
  # alignment should be.
  similarity /= temperature
  
  return similarity


def align_pair_of_sequences(embs1,
                            embs2,
                            similarity_type,
                            temperature):
  """Align a given pair embedding sequences.
  Args:
    embs1: Tensor, Embeddings of the shape [M, D] where M is the number of
      embeddings and D is the embedding size.
    embs2: Tensor, Embeddings of the shape [N, D] where N is the number of
      embeddings and D is the embedding size.
    similarity_type: String, Either one of 'l2' or 'cosine'.
    temperature: Float, Temperature used in scaling logits before softmax.
  Returns:
     logits: Tensor, Pre-softmax similarity scores after cycling back to the
      starting sequence.
    labels: Tensor, One hot labels containing the ground truth. The index where
      the cycle started is 1.
  """
  # max_num_steps is embs1's frame number
  max_num_steps = tf.shape(embs1)[0]

  # Find distances between embs1 and embs2.
  sim_12 = get_scaled_similarity(embs1, embs2, similarity_type, temperature)
  
  # Softmax the distance.
  softmaxed_sim_12 = tf.nn.softmax(sim_12, axis=1)
  # softmaxed_sim_12 is alpha in TCC paper
  # Calculate soft-nearest neighbors.
  nn_embs = tf.matmul(softmaxed_sim_12, embs2)
  # Find distances between nn_embs and embs1.
  sim_21 = get_scaled_similarity(nn_embs, embs1, similarity_type, temperature)
  logits = sim_21
  labels = tf.one_hot(tf.range(max_num_steps), max_num_steps)

  return logits, labels

def _align_single_cycle(cycle, embs, cycle_length, num_steps,
                        similarity_type, temperature):
  """Takes a single cycle and returns logits (simialrity scores) and labels."""
  # Choose random frame.
  n_idx = tf.random.uniform((), minval=0, maxval=num_steps, dtype=tf.int32)
  # Create labels
  onehot_labels = tf.one_hot(n_idx, num_steps)

  # Choose query feats for first frame.
  query_feats = embs[cycle[0], n_idx:n_idx+1]

  num_channels = tf.shape(query_feats)[-1]
  for c in range(1, cycle_length+1):
    candidate_feats = embs[cycle[c]]

    if similarity_type == 'l2':
      # Find L2 distance.
      mean_squared_distance = tf.reduce_sum(
          tf.square(tf.tile(query_feats, [num_steps, 1])- candidate_feats), axis=1)
      # Convert L2 distance to similarity.
      similarity = -mean_squared_distance

    elif similarity_type == 'cosine':
      # Dot product of embeddings.
      similarity = tf.squeeze(tf.matmul(candidate_feats, query_feats,
                                        transpose_b=True))
    else:
      raise ValueError('similarity_type can either be l2 or cosine.')

    # Scale the distance  by number of channels. This normalization helps with
    # optimization.
    similarity = tf.truediv(similarity,
                            tf.cast(num_channels, tf.float32))
    # # Scale the distance by a temperature that helps with how soft/hard the
    # # alignment should be.
    similarity = tf.truediv(similarity, temperature)

    beta = tf.nn.softmax(similarity)
    beta = tf.expand_dims(beta, axis=1)
    beta = tf.tile(beta, [1, num_channels])

    # Find weighted nearest neighbour.
    query_feats = tf.reduce_sum(beta * candidate_feats,
                                axis=0, keepdims=True)

  return similarity, onehot_labels


def _align(cycles, embs, num_steps, num_cycles, cycle_length,
           similarity_type, temperature):
  """Align by finding cycles in embs."""
  logits_list = []
  labels_list = []
  for i in range(num_cycles):
    logits, labels = _align_single_cycle(cycles[i],
                                         embs,
                                         cycle_length,
                                         num_steps,
                                         similarity_type,
                                         temperature)
    logits_list.append(logits)
    labels_list.append(labels)

  logits = tf.stack(logits_list)
  labels = tf.stack(labels_list)

  return logits, labels


def gen_cycles(num_cycles, batch_size, cycle_length=2):
  """Generates cycles for alignment.
  Generates a batch of indices to cycle over. For example setting num_cycles=2,
  batch_size=5, cycle_length=3 might return something like this:
  cycles = [[0, 3, 4, 0], [1, 2, 0, 3]]. This means we have 2 cycles for which
  the loss will be calculated. The first cycle starts at sequence 0 of the
  batch, then we find a matching step in sequence 3 of that batch, then we
  find matching step in sequence 4 and finally come back to sequence 0,
  completing a cycle.
  Args:
    num_cycles: Integer, Number of cycles that will be matched in one pass.
    batch_size: Integer, Number of sequences in one batch.
    cycle_length: Integer, Length of the cycles. If we are matching between
      2 sequences (cycle_length=2), we get cycles that look like [0,1,0].
      This means that we go from sequence 0 to sequence 1 then back to sequence
      0. A cycle length of 3 might look like [0, 1, 2, 0].
  Returns:
    cycles: Tensor, Batch indices denoting cycles that will be used for
      calculating the alignment loss.
  """
  sorted_idxes = tf.tile(tf.expand_dims(tf.range(batch_size), 0),
                         [num_cycles, 1])
  sorted_idxes = tf.reshape(sorted_idxes, [batch_size, num_cycles])
  cycles = tf.reshape(tf.random.shuffle(sorted_idxes),
                      [num_cycles, batch_size])
  cycles = cycles[:, :cycle_length]
  # Append the first index at the end to create cycle.
  cycles = tf.concat([cycles, cycles[:, 0:1]], axis=1)
  return cycles


def compute_stochastic_alignment_loss(embs,
                                      steps,
                                      seq_lens,
                                      num_steps,
                                      batch_size,
                                      loss_type,
                                      similarity_type,
                                      num_cycles,
                                      cycle_length,
                                      temperature,
                                      label_smoothing,
                                      variance_lambda,
                                      huber_delta,
                                      normalize_indices):
  """Compute cycle-consistency loss by stochastically sampling cycles.
  Args:
    embs: Tensor, sequential embeddings of the shape [N, T, D] where N is the
      batch size, T is the number of timesteps in the sequence, D is the size of
      the embeddings.
    steps: Tensor, step indices/frame indices of the embeddings of the shape
      [N, T] where N is the batch size, T is the number of the timesteps.
    seq_lens: Tensor, Lengths of the sequences from which the sampling was done.
      This can provide additional information to the alignment loss.
    num_steps: Integer/Tensor, Number of timesteps in the embeddings.
    batch_size: Integer/Tensor, Batch size.
    loss_type: String, This specifies the kind of loss function to use.
      Currently supported loss functions: 'classification', 'regression_mse',
      'regression_mse_var', 'regression_huber'.
    similarity_type: String, Currently supported similarity metrics: 'l2',
      'cosine'.
    num_cycles: Integer, number of cycles to match while aligning
      stochastically.  Only used in the stochastic version.
    cycle_length: Integer, Lengths of the cycle to use for matching. Only used
      in the stochastic version. By default, this is set to 2.
    temperature: Float, temperature scaling used to scale the similarity
      distributions calculated using the softmax function.
    label_smoothing: Float, Label smoothing argument used in
      tf.keras.losses.categorical_crossentropy function and described in this
      paper https://arxiv.org/pdf/1701.06548.pdf.
    variance_lambda: Float, Weight of the variance of the similarity
      predictions while cycling back. If this is high then the low variance
      similarities are preferred by the loss while making this term low results
      in high variance of the similarities (more uniform/random matching).
    huber_delta: float, Huber delta described in tf.keras.losses.huber_loss.
    normalize_indices: Boolean, If True, normalizes indices by sequence lengths.
      Useful for ensuring numerical instabilities doesn't arise as sequence
      indices can be large numbers.
  Returns:
    loss: Tensor, Scalar loss tensor that imposes the chosen variant of the
      cycle-consistency loss.
  """
  # Generate cycles.
  cycles = gen_cycles(num_cycles, batch_size, cycle_length)

  logits, labels = _align(cycles, embs, num_steps, num_cycles, cycle_length,
                          similarity_type, temperature)

  if loss_type == 'classification':
    loss = classification_loss(logits, labels, label_smoothing)
  elif 'regression' in loss_type:
    steps = tf.gather(steps, cycles[:, 0])
    seq_lens = tf.gather(seq_lens, cycles[:, 0])
    loss = regression_loss(logits, labels, num_steps, steps, seq_lens,
                           loss_type, normalize_indices, variance_lambda,
                           huber_delta)
  else:
    raise ValueError('Unidentified loss type %s. Currently supported loss '
                     'types are: regression_mse, regression_huber, '
                     'classification .'
                     % loss_type)
  return loss


def compute_deterministic_alignment_loss(embs,
                                         steps,
                                         seq_lens,
                                         num_steps,
                                         batch_size,
                                         loss_type,
                                         similarity_type,
                                         temperature,
                                         label_smoothing,
                                         variance_lambda,
                                         huber_delta,
                                         normalize_indices):
  """Compute cycle-consistency loss for all steps in each sequence.
  This aligns each pair of videos in the batch except with itself.
  When aligning it also matters which video is the starting video. So for N
  videos in the batch, we have N * (N-1) alignments happening.
  For example, a batch of size 3 has 6 pairs of sequence alignments.
  Args:
    embs: Tensor, sequential embeddings of the shape [N, T, D] where N is the
      batch size, T is the number of timesteps in the sequence, D is the size
      of the embeddings.
    steps: Tensor, step indices/frame indices of the embeddings of the shape
      [N, T] where N is the batch size, T is the number of the timesteps.
    seq_lens: Tensor, Lengths of the sequences from which the sampling was
    done. This can provide additional information to the alignment loss.
    num_steps: Integer/Tensor, Number of timesteps in the embeddings.
    batch_size: Integer, Size of the batch.
    loss_type: String, This specifies the kind of loss function to use.
      Currently supported loss functions: 'classification', 'regression_mse',
      'regression_mse_var', 'regression_huber'.
    similarity_type: String, Currently supported similarity metrics: 'l2' ,
      'cosine' .
    temperature: Float, temperature scaling used to scale the similarity
      distributions calculated using the softmax function.
    label_smoothing: Float, Label smoothing argument used in
      tf.keras.losses.categorical_crossentropy function and described in this
      paper https://arxiv.org/pdf/1701.06548.pdf.
    variance_lambda: Float, Weight of the variance of the similarity
      predictions while cycling back. If this is high then the low variance
      similarities are preferred by the loss while making this term low
      results in high variance of the similarities (more uniform/random
      matching).
    huber_delta: float, Huber delta described in tf.keras.losses.huber_loss.
    normalize_indices: Boolean, If True, normalizes indices by sequence
      lengths. Useful for ensuring numerical instabilities doesn't arise as
      sequence indices can be large numbers.
  Returns:
    loss: Tensor, Scalar loss tensor that imposes the chosen variant of the
        cycle-consistency loss.
  """
  labels_list = []
  logits_list = []
  steps_list = []
  seq_lens_list = []

  for i in range(batch_size):
    for j in range(batch_size):
      # We do not align the sequence with itself.
      if i != j:
        logits, labels = align_pair_of_sequences(embs[i],
                                                 embs[j],
                                                 similarity_type,
                                                 temperature)
        logits_list.append(logits)
        labels_list.append(labels)
        steps_list.append(tf.tile(steps[i:i+1], [num_steps, 1]))
        seq_lens_list.append(tf.tile(seq_lens[i:i+1], [num_steps]))

  logits = tf.concat(logits_list, axis=0)
  labels = tf.concat(labels_list, axis=0)
  steps = tf.concat(steps_list, axis=0)
  seq_lens = tf.concat(seq_lens_list, axis=0)

  if loss_type == 'classification':
    loss = classification_loss(logits, labels, label_smoothing)
  elif 'regression' in loss_type:
    loss = regression_loss(logits, labels, num_steps, steps, seq_lens,
                           loss_type, normalize_indices, variance_lambda,
                           huber_delta)
  else:
    raise ValueError('Unidentified loss_type %s. Currently supported loss '
                     'types are: regression_mse, regression_huber, '
                     'classification.' % loss_type)

  return loss