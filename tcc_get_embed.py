import tensorflow as tf
import numpy as np
from tcc_config import CONFIG
import datetime
import os

from absl import app
from absl import flags

# utils
import utils.tcc_data as tccdata
import utils.tcc_embed as tccembed

# Specify GPUs
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"

flags.DEFINE_string('dataset', None, 'Dataset of the processing action.')
flags.DEFINE_string('mode', 'val', 'train or val')
FLAGS = flags.FLAGS

flags.mark_flag_as_required('dataset')

def get_embed():
  model = tccembed.Embedder(CONFIG.EMBEDDING_SIZE, CONFIG.NORMALIZE_EMBEDDINGS, CONFIG.NUM_CONTEXT_STEPS)
  optimizer = tf.keras.optimizers.Adam(CONFIG.LEARNING_RATE) 
  ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
  manager = tf.train.CheckpointManager(ckpt, CONFIG.LOGDIR, max_to_keep=3)
  ckpt.restore(manager.latest_checkpoint)

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

  # Extract per-frame embeddings
  print("Extracting per-frame embeddings...")
  embs = tccembed.get_embs(model, videos, video_seq_lens,
                  frames_per_batch=CONFIG.FRAMES_PER_BATCH, 
                  num_context_steps=CONFIG.NUM_CONTEXT_STEPS,
                  context_stride=CONFIG.CONTEXT_STRIDE)

  # Save the embeddings so that you don't have to use GPU for later experiments.
  path_to_embs = CONFIG.PATH_TO_EMBS
  np.save(path_to_embs % FLAGS.dataset, embs)
  print('Embeddings saved.')

def main(_argv):
  get_embed()

if __name__ == '__main__':
  try:
    app.run(main)
  except SystemExit:
    pass