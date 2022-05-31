import scipy.io as sio
import numpy as np
import cv2
import glob
import os
import json
'''
# Load .mat file
LABEL_PATH = './data/Penn_Action/labels'

for mat_file in sorted(os.listdir(LABEL_PATH)):
    mat_contents = sio.loadmat(os.path.join(LABEL_PATH, mat_file))
    print(mat_file.split('.')[0])
    now = mat_contents['action'][0]
    #print(mat_contents['action'])
    #print(mat_contents['pose'])
    #print(np.shape(mat_contents['bbox']))
    #print(mat_contents.keys())

with open("./data/Penn_Action/video_lists/tennis_forehand_val.txt", "w") as file:
  for i in range(2069,2141):
    file.write('{0:04d}\n'.format(i))
'''
'''
# Create JSON file
#with open('./data/Penn_Label.json', 'r',  newline='') as jsonfile:
#  data = json.load(jsonfile)
with open('./data/Penn_Label.json', 'w', newline='') as jsonfile:
  data = []
  for mat_file in sorted(os.listdir(LABEL_PATH)):
    mat_contents = sio.loadmat(os.path.join(LABEL_PATH, mat_file))
    data.append({
        'id': mat_file.split('.')[0],
        'action': mat_contents['action'][0],
    })
  json.dump(data, jsonfile)
'''



'''
# Load Frames
def pad_zeros(frames, max_seq_len):
  npad = ((0, max_seq_len-len(frames)), (0, 0), (0, 0), (0, 0))
  frames = np.pad(frames, pad_width=npad, mode='constant', constant_values=0)
  return frames

# TODO Need to import action label to seperate actions
video_foldernames = sorted(glob.glob(os.path.join('./data/Penn_Action_Test/', 'frames/*')))
print('Found %d videos.'%len(video_foldernames))
videos = []
video_seq_lens = []
for video_foldername in video_foldernames:
    print('Loading frames in' + video_foldername)
    for filename in os.listdir(video_foldername):
        frames = cv2.imread(os.path.join(video_foldername, filename))
        videos.append(frames)
        video_seq_lens.append(len(frames))
max_seq_len = max(video_seq_lens)
videos = np.asarray([pad_zeros(x, max_seq_len) for x in videos])


# List of Labels
# ['__header__', '__version__', '__globals__', 'action', 'pose',
#  'x', 'y', 'visibility', 'train', 'bbox', 'dimensions', 'nframes']

# 'action'     : (string)  Action label for this video.
# 'pose'       : (string)  Viewpoint of this video.
# 'x', 'y'     : (double)  Joints location in each frame. Shape = nframes*13
# 'visibility' : (boolean) Visibility for each joint in each frame. Shape = nframes*13
# 'train'      : (1 or -1) 1: train, -1: test
# 'bbox'       : (double)  Human bounding box in each frame. Shape = nframes*4
# 'dimensions' : (int)     Video dimension. [Height Width nframes]
# 'nframes'    : (int)     Video frame number.

# List of Actions
# baseball_pitch  clean_and_jerk  pull_ups  strumming_guitar  
# baseball_swing  golf_swing      push_ups  tennis_forehand   
# bench_press     jumping_jacks   sit_ups   tennis_serve
# bowling         jump_rope       squats

# List of Annotated Joints
# 1.  head       
# 2.  left_shoulder  3.  right_shoulder
# 4.  left_elbow     5.  right_elbow
# 6.  left_wrist     7.  right_wrist     
# 8.  left_hip       9.  right_hip 
# 10. left_knee      11. right_knee 
# 12. left_ankle     13. right_ankle
'''