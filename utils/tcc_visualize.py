from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
from dtw import *
import math
from sklearn import manifold

# Visualization code
def dist_fn(x, y):
  dist = np.sum((x-y)**2)
  return dist


def get_nn(embs, query_emb):
  dist = np.linalg.norm(embs - query_emb, axis=1)
  assert len(dist) == len(embs)
  return np.argmin(dist), np.min(dist)


def unnorm(query_frame):
  min_v = query_frame.min()
  max_v = query_frame.max()
  query_frame = (query_frame - min_v) / (max_v - min_v)
  return query_frame


def viz_align(query_feats, candidate_feats, use_dtw):
  """Align videos based on dynamic time warping."""
  if use_dtw:
    print('[CLEA] candidate_feats length = ', len(candidate_feats))
    # dtw() returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    # min_dist represents the similarity of two sequences.
    min_dist, cost_matrix, acc_cost_matrix, path = dtw(query_feats, candidate_feats, dist=dist_fn)
    print('[CLEA] min_dist = ', min_dist)
    #print('[CLEA] cost_matrix = ', cost_matrix)
    #print('[CLEA] acc_cost_matrix = ', acc_cost_matrix)
    #print('[CLEA] path = ', path)
    _, uix = np.unique(path[0], return_index=True) # uix is the index of the unique element
    nns = path[1][uix]
  else:
    nns = []
    for i in range(len(query_feats)):
      nn_frame_id, _ = get_nn(candidate_feats, query_feats[i])
      nns.append(nn_frame_id)
  return nns

def show_video(video_path):
  mp4 = open(video_path,'rb').read()
  data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
  return HTML("""<video width=600 controls>
      <source src="%s" type="video/mp4"></video>
  """ % data_url)


def create_video(embs, frames, video_path, use_dtw, query=0):
  """Create aligned videos."""
  # If candiidate is not None implies alignment is being calculated between
  # 2 videos only.
  
  for i in range(len(frames)):
    print('frame[{}] shape = {}'.format(i, frames[i].shape))

  ncols = int(math.sqrt(len(embs)))
  fig, ax = plt.subplots(
      ncols=ncols,
      nrows=ncols,
      figsize=(5 * ncols, 5 * ncols),
      tight_layout=True)

  nns = []
  for candidate in range(len(embs)):
    nns.append(viz_align(embs[query], embs[candidate], use_dtw))
  ims = []

  def init():
    k = 0
    for k in range(ncols):
      for j in range(ncols):
        ims.append(ax[j][k].imshow(
            unnorm(frames[k * ncols + j][nns[k * ncols + j][0]])))
        ax[j][k].grid(False)
        ax[j][k].set_xticks([])
        ax[j][k].set_yticks([])
        ax[j][k].set_title('ax[{}][{}]'.format(j,k), color=plt.cm.Set1(k * ncols + j), fontsize = 14)
    return ims

  ims = init()

  def update(i):
    for k in range(ncols):
      for j in range(ncols):
        ims[k * ncols + j].set_data(
            unnorm(frames[k * ncols + j][nns[k * ncols + j][i]]))
        ax[j][k].set_title('FRAME {}'.format(nns[k * ncols + j][i]), color=plt.cm.Set1(k * ncols + j), fontsize = 14)
    plt.tight_layout()
    return ims

  anim = FuncAnimation(
      fig,
      update,
      frames=np.arange(len(embs[query])),
      interval=100,
      blit=False)
  anim.save(video_path, dpi=40)

  plt.close()

def create_dynamic_video(embs, frames, video_path, use_dtw, query=0):
  """Create aligned videos."""
  fig, ax = plt.subplots(ncols=2, figsize=(10, 5), tight_layout=True)

  ax[0].set_title('Reference Frame')
  ax[1].set_title('Aligned Frame using TCC')
  nns = []
  for candidate in range(len(embs)):
    nns.append(viz_align(embs[query], embs[candidate], use_dtw))

  switch_video = max(1, len(embs[query])//len(embs))

  im0 = ax[0].imshow(unnorm(frames[0][0]))
  im1 = ax[1].imshow(unnorm(frames[1][nns[1][0]]))

  def update(i):
    """Update plot with next frame."""
    candidate = min(i // switch_video + 1,
                    len(embs)-1)

    im0.set_data(unnorm(frames[query][i]))
    im1.set_data(unnorm(frames[candidate][nns[candidate][i]]))
    # Hide grid lines
    ax[0].grid(False)
    ax[1].grid(False)

    # Hide axes ticks
    ax[0].set_xticks([])
    ax[1].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_yticks([])
    plt.tight_layout()

  anim = FuncAnimation(
      fig,
      update,
      frames=np.arange(len(embs[query])),
      interval=100,
      blit=False)
  anim.save(video_path, dpi=80)
  plt.close()


def viz_alignment(embs,
                  frames,
                  video_path,
                  grid_mode=True,
                  use_dtw=False,
                  query=0):
  """Visualize alignment."""

  if grid_mode:
    return create_video(
        embs,
        frames,
        video_path,
        use_dtw,
        query)
  else:
    return create_dynamic_video(
        embs,
        frames,
        video_path,
        use_dtw,
        query)

def viz_tSNE(embs,
            frames,
            output_path,
            use_dtw=False,
            query=0):
  nns = []
  idx = np.arange(len(embs))
  for candidate in range(len(embs)):
    idx[candidate] = candidate
    nns.append(viz_align(embs[query], embs[candidate], use_dtw))
  
  X = np.empty((0, 128))
  y = []
  frame_idx = []
  for i, video_emb in zip(idx, embs):
    for j in range(len(embs[0])):
      X = np.append(X, np.array([video_emb[nns[i][j]]]), axis=0)
      y.append(int(i))
      frame_idx.append(j)
  y = np.array(y)
  frame_idx = np.array(frame_idx)

  #t-SNE
  X_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(X)

  #Data Visualization
  x_min, x_max = X_tsne.min(0), X_tsne.max(0)
  X_norm = (X_tsne - x_min) / (x_max - x_min)  #Normalize
  plt.figure(figsize=(8, 8))
  for i in range(X_norm.shape[0]):
      plt.text(X_norm[i, 0], X_norm[i, 1], str(frame_idx[i]), color=plt.cm.Set1(y[i]), 
              fontdict={'weight': 'bold', 'size': 9})
  plt.xticks([])
  plt.yticks([])
  plt.savefig(output_path)