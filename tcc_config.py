from easydict import EasyDict as edict
import os

# Get the configuration dictionary whose keys can be accessed with dot.
CONFIG = edict()

CONFIG.LOGDIR = '/home/clealin/tcc/tcc_colab/log/'
#CONFIG.PATH_TO_RAW_VIDEOS = '/home/clealin/tcc/tcc_colab/data/Penn_Action/datasets_raw/'
#CONFIG.PATH_TO_RAW_VIDEOS = '/home/clealin/tcc/tcc_colab/data/Penn_Action/'
CONFIG.PATH_TO_RAW_VIDEOS = '/home/clealin/tcc/tcc_colab/data/Skating/'

CONFIG.BATCH_SIZE =  2#@param {type:"integer"}
CONFIG.NUM_STEPS = 40 #@param {type:"integer"}
CONFIG.NUM_CONTEXT_STEPS =  2#@param {type:"integer"}
CONFIG.CONTEXT_STRIDE =  15#@param {type:"integer"}

CONFIG.LOSS_TYPE = 'regression_mse_var' #@param ["regression_mse_var", "regression_mse", "regression_huber", "classification"]
CONFIG.STOCHASTIC_MATCHING = False #@param ["False", "True"] {type:"raw"}
CONFIG.TCC_TCN_COMBINE = False
CONFIG.SIMILARITY_TYPE = 'l2' #@param ["l2", "cosine"]
CONFIG.EMBEDDING_SIZE =  128 #@param {type:"integer"}
CONFIG.TEMPERATURE = 0.1 #@param {type:"number"}
CONFIG.LABEL_SMOOTHING = 0.0 #@param {type:"slider", min:0, max:1, step:0.05}                                   
CONFIG.VARIANCE_LAMBDA = 0.001 #@param {type:"number"}                                       
CONFIG.HUBER_DELTA = 0.1 #@param {type:"number"}                                        
CONFIG.NORMALIZE_INDICES = True #@param ["False", "True"] {type:"raw"}
CONFIG.NORMALIZE_EMBEDDINGS = False #@param ["False", "True"] {type:"raw"}

CONFIG.CYCLE_LENGTH = 2 #@param {type:"integer"}
CONFIG.NUM_CYCLES = 32 #@param {type:"integer"}

CONFIG.LEARNING_RATE = 1e-4 #@param {type:"number"}
CONFIG.DEBUG = False #@param ["False", "True"] {type:"raw"}

# Training parameter
CONFIG.MAX_NUM_TRAINING_STEPS = 1000
CONFIG.SAVE_CKPT_STEPS = 200
CONFIG.SHOW_LOSS_STEPS = 1

# Extract embedding parameter
CONFIG.PATH_TO_EMBS = os.path.join(CONFIG.LOGDIR, '%s_embeddings.npy')
CONFIG.FRAMES_PER_BATCH = 160 # Change if you have more GPU memory.

CONFIG.KENDALLS_TAU_STRIDE = 2  # Compare n frame at a time. 5 for Pouring, 2 for PennAction
CONFIG.KENDALLS_TAU_DISTANCE = 'sqeuclidean'  # cosine, sqeuclidean

# ******************************************************************************
# Time Contrastive Network params
# ******************************************************************************
CONFIG.REG_LAMBDA = 0.002

# Skeleton parameters
CONFIG.SKELETON_CONF_THRESHOLD = 0.0