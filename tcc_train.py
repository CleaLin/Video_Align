import tensorflow as tf
from tcc_config import CONFIG
import os
import datetime

from absl import app
from absl import flags

# utils
import utils.tcc_data as tccdata
import utils.tcc_embed as tccembed
import utils.tcc_loss as tccloss
import utils.tcn_loss as tcnloss

# Specify GPUs
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6"

flags.DEFINE_string('dataset', None, 'Dataset of the processing action.')
flags.DEFINE_string('mode', 'train', 'train or val')
FLAGS = flags.FLAGS

flags.mark_flag_as_required('dataset')

def train():
  # Load videos and get all frames
  if 'pouring' in FLAGS.dataset:
    videos, video_seq_lens = tccdata.load_videos(CONFIG.PATH_TO_RAW_VIDEOS, FLAGS.dataset, FLAGS.mode)
  elif 'skate' in FLAGS.dataset:
    videos, video_seq_lens, _, _ = tccdata.load_skate_data(CONFIG.PATH_TO_RAW_VIDEOS, FLAGS.dataset, FLAGS.mode)
  else:
    videos, video_seq_lens = tccdata.load_penn_data(CONFIG.PATH_TO_RAW_VIDEOS, FLAGS.dataset, FLAGS.mode)
  print('------------------------------------------------------')
  print('-----------------Finish loading data.-----------------')
  print('------------------------------------------------------')
  # Create training dataset
  train_dataset = tccdata.create_dataset(videos, video_seq_lens,
                          batch_size=CONFIG.BATCH_SIZE,
                          num_steps=CONFIG.NUM_STEPS,
                          num_context_steps=CONFIG.NUM_CONTEXT_STEPS,
                          context_stride=CONFIG.CONTEXT_STRIDE)
  # Create model
  model = tccembed.Embedder(CONFIG.EMBEDDING_SIZE, CONFIG.NORMALIZE_EMBEDDINGS, CONFIG.NUM_CONTEXT_STEPS)
  optimizer = tf.keras.optimizers.Adam(CONFIG.LEARNING_RATE)
  learning_rate = optimizer.learning_rate
  ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
  manager = tf.train.CheckpointManager(ckpt, CONFIG.LOGDIR, max_to_keep=3)
  summary_writer = tf.summary.create_file_writer(CONFIG.LOGDIR, flush_millis=1000)

  # Create training loop
  @tf.function
  def train_one_iter(data):
    frames = data['frames']
    steps = data['steps']
    seq_lens = data['seq_lens']
    with tf.GradientTape() as tape:
      embs = model(frames, training=True)
      trainable_variables = model.trainable_variables
      if CONFIG.STOCHASTIC_MATCHING:
        loss = tccloss.compute_stochastic_alignment_loss(embs,
                                          steps,
                                          seq_lens,
                                          num_cycles=CONFIG.NUM_CYCLES,
                                          cycle_length=CONFIG.CYCLE_LENGTH,
                                          num_steps=CONFIG.NUM_STEPS,
                                          batch_size=CONFIG.BATCH_SIZE,
                                          loss_type=CONFIG.LOSS_TYPE,
                                          similarity_type=CONFIG.SIMILARITY_TYPE,
                                          temperature=CONFIG.TEMPERATURE,
                                          label_smoothing=CONFIG.LABEL_SMOOTHING,
                                          variance_lambda=CONFIG.VARIANCE_LAMBDA,
                                          huber_delta=CONFIG.HUBER_DELTA,
                                          normalize_indices=CONFIG.NORMALIZE_INDICES)
      else:
          loss = tccloss.compute_deterministic_alignment_loss(embs,
                                          steps,
                                          seq_lens,
                                          num_steps=CONFIG.NUM_STEPS,
                                          batch_size=CONFIG.BATCH_SIZE,
                                          loss_type=CONFIG.LOSS_TYPE,
                                          similarity_type=CONFIG.SIMILARITY_TYPE,
                                          temperature=CONFIG.TEMPERATURE,
                                          label_smoothing=CONFIG.LABEL_SMOOTHING,
                                          variance_lambda=CONFIG.VARIANCE_LAMBDA,
                                          huber_delta=CONFIG.HUBER_DELTA,
                                          normalize_indices=CONFIG.NORMALIZE_INDICES)
      
      if CONFIG.TCC_TCN_COMBINE:
        tcn_loss = tcnloss.compute_tcn_loss(embs, FLAGS.mode == 'train')
        loss = loss*0.7+tcn_loss*0.3
      
      # Add regularization losses.
      if model.losses:
        loss += tf.add_n(model.losses)

    grads = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(grads, trainable_variables))

    with tf.summary.record_if(tf.math.equal(
        tf.math.mod(optimizer.iterations, 10), 0)):
      tf.summary.scalar('loss', loss, optimizer.iterations)
      if CONFIG.DEBUG:
        for var_ in model.variables:
          tf.summary.histogram(var_.name, var_, step=optimizer.iterations)
        _, n, h, w, c = frames.shape
        tf.summary.image('frames', tf.cast(
          255.0*(tf.squeeze(tf.concat(tf.split(frames, n, axis=1),
                                      axis=3), axis=1)+1.0)/2.0, tf.uint8),
                        optimizer.iterations)

    return loss

  # Train the model
  # Uncomment this to load previous checkpoint.
  #ckpt.restore(manager.latest_checkpoint)
  tf.keras.backend.set_learning_phase(1)
  i = 1
  with summary_writer.as_default():
    for data in train_dataset.take(CONFIG.MAX_NUM_TRAINING_STEPS):
      loss = train_one_iter(data)
      if i % CONFIG.SHOW_LOSS_STEPS == 0:
        now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")[:-3]
        print("{now_time} ITER:{step_now}/{step_max}, Loss: {loss_now:.3f}".format(now_time=now_time, step_now=i, step_max=CONFIG.MAX_NUM_TRAINING_STEPS, loss_now=loss))
      if i % CONFIG.SAVE_CKPT_STEPS == 0:
        manager.save()
      i += 1

  # Save model if you inerrupted training or finished training.
  manager.save()

def main(_argv):
  train()

if __name__ == '__main__':
  try:
    app.run(main)
  except SystemExit:
    pass