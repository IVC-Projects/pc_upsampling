import argparse
import os
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import socket
import Code.model_gen as MODEL_GEN
import Code.model_utils as model_utils
import Code.data_provider as data_provider
from Utils import pc_util
from Code import prediction_layer

parser = argparse.ArgumentParser()
parser.add_argument('--phase', default='test', help='train or test [default: train]')
parser.add_argument('--gpu', default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='../model', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=256,help='Point Number [1024/2048] [default: 1024]')
parser.add_argument('--up_ratio',  type=int,  default=4,  help='Upsampling Ratio [default: 2]')
parser.add_argument('--max_epoch', type=int, default=120, help='Epoch to run [default: 500]')
parser.add_argument('--batch_size', type=int, default=48, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001)

ASSIGN_MODEL_PATH=None
USE_RANDOM_INPUT = False
USE_REPULSION_LOSS = True

FLAGS = parser.parse_args()
PHASE = FLAGS.phase
GPU_INDEX = FLAGS.gpu
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
UP_RATIO = FLAGS.up_ratio
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
MODEL_DIR = FLAGS.log_dir

print(socket.gethostname())
print(FLAGS)

os.environ['CUDA_VISIBLE_DEVICES'] = GPU_INDEX

def log_string(out_str):
    global LOG_FOUT
    LOG_FOUT.write(out_str)
    LOG_FOUT.flush()

def prediction_whole_model():
    data_folder = '../data/test_data/MC_input'
    with tf.Session() as sess:
        prediction_layer.prediction(sess, data_folder, 256, MODEL_GEN, MODEL_DIR, UP_RATIO).build()


if __name__ == "__main__":
    np.random.seed(int(time.time()))
    tf.set_random_seed(int(time.time()))
    prediction_whole_model()
