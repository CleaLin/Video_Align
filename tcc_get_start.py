import tensorflow as tf
from tcc_config import CONFIG
import os
import numpy as np
from dtw import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import datetime

from absl import app
from absl import flags

# utils
import utils.tcc_data as tccdata
import utils.tcc_embed as tccembed
import utils.tcc_skeleton as tccskeleton

# Specify GPUs
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6"

flags.DEFINE_string('dataset', None, 'Dataset of the processing action.')
flags.DEFINE_string('mode', 'val', 'train or val')
FLAGS = flags.FLAGS

flags.mark_flag_as_required('dataset')

def get_start(query=0, candi=1):
  model = tccembed.Embedder(CONFIG.EMBEDDING_SIZE, CONFIG.NORMALIZE_EMBEDDINGS, CONFIG.NUM_CONTEXT_STEPS)
  optimizer = tf.keras.optimizers.Adam(CONFIG.LEARNING_RATE) 
  ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
  manager = tf.train.CheckpointManager(ckpt, CONFIG.LOGDIR, max_to_keep=3)
  ckpt.restore(manager.latest_checkpoint)

  # Load videos and get all frames
  if 'pouring' in FLAGS.dataset:
    videos, video_seq_lens = tccdata.load_videos(CONFIG.PATH_TO_RAW_VIDEOS, FLAGS.dataset, FLAGS.mode)
  elif 'skate' in FLAGS.dataset:
    videos, video_seq_lens, videos_raw, skeletons = tccdata.load_skate_data(CONFIG.PATH_TO_RAW_VIDEOS, FLAGS.dataset, FLAGS.mode)
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
  
  def dist_fn(x, y):
    dist = np.sum((x-y)**2)
    return dist

  # Slide candi(embs[1]) through each frame in query(embs[0])
  # Default embs[0] as the longer video and embs[1] as the standard video
  min_dists = []
  for i in range(len(embs[query])-len(embs[candi])):
    query_embs = embs[query][i:i+len(embs[candi])]
    candidate_embs = embs[candi]
    min_dist, cost_matrix, acc_cost_matrix, path = dtw(query_embs, candidate_embs, dist=dist_fn)
    min_dists.append(min_dist)
  
  start_frame = min_dists.index(min(min_dists))
  print('[CLEA] START FRAME = ', start_frame)
  x = np.arange(0, len(embs[query])-len(embs[candi]))
  plt.plot(x, min_dists, '-ro', markevery=[start_frame])
  plt.savefig('min_dists')

  # Draw skeleton on raw videos
  videos_drawn = tccskeleton.vis_skeleton(videos_raw, skeletons)

  return start_frame, videos_drawn

# Create video
def gen_result(start_frame, frames, query=0, candi=1):
  # Draw skeleton on query video

  now_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  OUTPUT_PATH = './result/align_{}.mp4'.format(now_time)
  
  # Create subplots
  nrows = len(frames)
  fig, ax = plt.subplots(
        ncols=nrows,
        figsize=(10 * nrows, 10 * nrows),
        tight_layout=True)
  
  def unnorm(query_frame):
    min_v = query_frame.min()
    max_v = query_frame.max()
    query_frame = (query_frame - min_v) / (max_v - min_v)
    return query_frame

  ims = []
  def init():
    k = 0
    for k in range(nrows):
      ims.append(ax[k].imshow(
          unnorm(frames[k][0])))
      ax[k].grid(False)
      ax[k].set_xticks([])
      ax[k].set_yticks([])
    return ims

  ims = init()

  def update(i):
    ims[query].set_data(unnorm(frames[query][i]))
    ax[query].set_title('FRAME {}'.format(i), fontsize = 14)
    if i >= start_frame and i < (start_frame+len(frames[candi])):
      ims[candi].set_data(unnorm(frames[candi][i-start_frame]))
    elif i < start_frame:
      ims[candi].set_data(unnorm(frames[candi][0]))
    else:
      ims[candi].set_data(unnorm(frames[candi][-1]))
    plt.tight_layout()
    return ims

  # Create animation
  anim = FuncAnimation(
      fig,
      update,
      frames=np.arange(len(frames[query])),
      interval=100,
      blit=False)
  anim.save(OUTPUT_PATH, dpi=40)

  plt.close()

def main(_argv):
  start_frame, org_frames = get_start()
  gen_result(start_frame, org_frames)

if __name__ == '__main__':
  try:
    app.run(main)
  except SystemExit:
    pass