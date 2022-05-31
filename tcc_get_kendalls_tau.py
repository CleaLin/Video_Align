import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import kendalltau
import tensorflow.compat.v2 as tf
from PIL import Image
import datetime

from tcc_config import CONFIG
from absl import app
from absl import flags

flags.DEFINE_string('dataset', None, 'Dataset of the processing action.')
flags.DEFINE_string('mode', 'val', 'train or val')
FLAGS = flags.FLAGS

flags.mark_flag_as_required('dataset')

def softmax(w, t=1.0):
  e = np.exp(np.array(w) / t)
  dist = e / np.sum(e)
  return dist

def get_kendalls_tau():
  """Get nearest neighbours in embedding space and calculate Kendall's Tau."""
  # Load previously saved embeddings in case you have them stored.
  path_to_embs = CONFIG.PATH_TO_EMBS
  embs_list = np.load(path_to_embs % FLAGS.dataset, allow_pickle=True)
  num_seqs = len(embs_list)
  taus = np.zeros((num_seqs * (num_seqs - 1)))
  idx = 0
  stride = CONFIG.KENDALLS_TAU_STRIDE
  for i in range(num_seqs):
    query_feats = embs_list[i][::stride]
    for j in range(num_seqs):
      if i == j:
        continue
      candidate_feats = embs_list[j][::stride]
      dists = cdist(query_feats, candidate_feats,
                    CONFIG.KENDALLS_TAU_DISTANCE)
      # Build similarity matrix for video 0 and 1
      '''
      if i == 0 and j == 1:
        sim_matrix = []
        for k in range(len(query_feats)):
          sim_matrix.append(softmax(-dists[k]))
        sim_matrix = np.array(sim_matrix, dtype=np.float32)
        sim_matrix = sim_matrix[None, :, :, None]
        sim_matrix = np.squeeze(sim_matrix)
        print('[CLEA] sim_matrix shape = ', sim_matrix.shape)
        now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        OUTPUT_PATH = '/home/clealin/tcc/tcc_colab/result/sim_matrix_{}.png'.format(now_time)
        img = Image.fromarray(sim_matrix, 'L')
        img.save(OUTPUT_PATH)
      '''
      nns = np.argmin(dists, axis=1)
      taus[idx] = kendalltau(np.arange(len(nns)), nns).correlation
      idx += 1
  # Remove NaNs.
  taus = taus[~np.isnan(taus)]
  tau = np.mean(taus)
  print('Kendalls Tau of {dataset}_{mode} is {tau:.4f}'.format(dataset=FLAGS.dataset, mode=FLAGS.mode, tau=tau))
  return tau

def main(_argv):
  get_kendalls_tau()

if __name__ == '__main__':
  try:
    app.run(main)
  except SystemExit:
    pass