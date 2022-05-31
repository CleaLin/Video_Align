import tensorflow as tf
import numpy as np
from tcc_config import CONFIG
import datetime

from absl import app
from absl import flags

# utils
import utils.tcc_data as tccdata
import utils.tcc_visualize as tccvisualize

flags.DEFINE_string('dataset', None, 'Dataset of the processing action.')
flags.DEFINE_string('mode', 'val', 'train or val')
FLAGS = flags.FLAGS

flags.mark_flag_as_required('dataset')

def align():
    # Load previously saved embeddings in case you have them stored.
    path_to_embs = CONFIG.PATH_TO_EMBS
    embs = np.load(path_to_embs % FLAGS.dataset, allow_pickle=True)

    # Load videos and get all frames
    if 'pouring' in FLAGS.dataset:
      videos, video_seq_lens = tccdata.load_videos(CONFIG.PATH_TO_RAW_VIDEOS, FLAGS.dataset, FLAGS.mode)
    elif 'skate' in FLAGS.dataset:
      videos, video_seq_lens, videos_raw = tccdata.load_skate_data(CONFIG.PATH_TO_RAW_VIDEOS, FLAGS.dataset, FLAGS.mode)
    else:
      videos, video_seq_lens = tccdata.load_penn_data(CONFIG.PATH_TO_RAW_VIDEOS, FLAGS.dataset, FLAGS.mode)
    print('------------------------------------------------------')
    print('-----------------Finish loading data.-----------------')
    print('------------------------------------------------------')
    
    # Randomly pick videos to align
    align_embs = []
    align_videos = []
    # Give each video an index
    video_index_pick = np.arange(0, len(videos_raw))
    np.random.shuffle(video_index_pick)
    for i in video_index_pick:
      align_embs.append(embs[i])
      align_videos.append(videos_raw[i])

    # Generate aligned video
    now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    print("Generating aligned video at /home/clealin/tcc/tcc_colab/result/output_{}.mp4".format(now_time))
    VIDEO_OUTPUT_PATH = '/home/clealin/tcc/tcc_colab/result/output_{}.mp4'.format(now_time)
    tSNE_OUTPUT_PATH = '/home/clealin/tcc/tcc_colab/result/output_{}.jpg'.format(now_time)
    NUM_VIDEOS = 4
    GRID_MODE = True
    USE_DTW = True
    
    tccvisualize.viz_alignment(align_embs[:NUM_VIDEOS],
                align_videos[:NUM_VIDEOS],
                VIDEO_OUTPUT_PATH,
                grid_mode=GRID_MODE,
                use_dtw=USE_DTW)
    
    tccvisualize.viz_tSNE(align_embs[:NUM_VIDEOS],
                align_videos[:NUM_VIDEOS],
                tSNE_OUTPUT_PATH,
                use_dtw=USE_DTW)

def main(_argv):
  align()

if __name__ == '__main__':
  try:
    app.run(main)
  except SystemExit:
    pass