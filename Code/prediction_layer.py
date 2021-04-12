import os
import tensorflow as tf
import numpy as np
from glob import glob
import Code.model_utils as model_utils
from Code import data_provider
from Code.data_provider import normalize_point_cloud
from tf_ops.sampling.tf_sampling import farthest_point_sample
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import time


def extract_knn_patch(queries, pc, k):
    """
    queries [M, C]
    pc [P, C]
    """
    knn_search = NearestNeighbors(n_neighbors=k, algorithm='auto')
    knn_search.fit(pc)
    knn_idx = knn_search.kneighbors(queries, return_distance=False)
    k_patches = np.take(pc, knn_idx, axis=0)  # M, K, C
    return k_patches

def patch_prediction(self, patch_point):
    # normalize the point clouds
    patch_point, centroid, furthest_distance = normalize_point_cloud(patch_point)
    patch_point = np.expand_dims(patch_point, axis=0)
    pred = self.sess.run([self.pred_pc], feed_dict={self.pointclouds_ipt: patch_point})
    pred = np.squeeze(centroid + pred * furthest_distance, axis=0)
    return pred

def pc_prediction(self, pc, patch_num_point=32, patch_num_ratio=3):
    ## get patch seed from farthestsampling
    points = tf.convert_to_tensor(np.expand_dims(pc, axis=0), dtype=tf.float32)
    seed1_num = int(pc.shape[0] / patch_num_point * patch_num_ratio)
    ## FPS sampling
    seed = farthest_point_sample(seed1_num, points).eval()[0]
    seed_list = seed[:seed1_num]
    print("number of patches: %d" % len(seed_list))
    up_point_list = []
    patches = extract_knn_patch(pc[np.asarray(seed_list), :], pc, patch_num_point)
    for point in tqdm(patches, total=len(patches)):
        up_point = patch_prediction(self, point)
        up_point_list.append(up_point)
    return up_point_list


class prediction(object):
    def __init__(self, sess, data_folder, patch_num, MODEL_GEN, MODEL_DIR, UP_RATIO):
        self.sess = sess
        self.data_folder = data_folder
        self.patch_num = patch_num
        self.MODEL_GEN = MODEL_GEN
        self.MODEL_DIR = MODEL_DIR
        self.UP_RATIO = UP_RATIO

    def build(self):
        phase = self.data_folder.split('/')[-2] + self.data_folder.split('/')[-1]
        save_path = os.path.join(self.MODEL_DIR, 'result/' + phase)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        samples = glob(self.data_folder + "/*.xyz")
        samples.sort(reverse=True)
        self.pointclouds_ipt = tf.placeholder(tf.float32, shape=(1, self.patch_num, 3))
        self.pred_pc, _ = self.MODEL_GEN.get_gen_model(self.pointclouds_ipt, is_training=False, scope='generator',
                                            reuse=None, use_bn=False, bn_decay=None, up_ratio=self.UP_RATIO)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        saver = tf.train.Saver()
        _, restore_model_path = model_utils.pre_load_checkpoint(self.MODEL_DIR)
        print(restore_model_path)
        saver.restore(self.sess, restore_model_path)
        samples = glob(self.data_folder + "/*.xyz")
        samples.sort()

        for i, item in enumerate(samples):
            start = time.time()
            input = np.loadtxt(item)
            input, centroid, furthest_distance = data_provider.normalize_point_cloud(input)
            pred_list = pc_prediction(self, input, self.patch_num)
            pred_pc = np.concatenate(pred_list, axis=0)
            pred_pc = (pred_pc * furthest_distance) + centroid
            pred_pc = np.reshape(pred_pc, [-1, 3])
            idx = farthest_point_sample(input.shape[0] * self.UP_RATIO, pred_pc[np.newaxis, ...]).eval()[0]
            pred_pc = pred_pc[idx, 0:3]
            end = time.time()
            print("time use:")
            print(end - start)
            path = os.path.join(save_path, item.split('/')[-1])
            print(path[:-4])
            print("================")
            np.savetxt(path[:-4] + '.xyz', pred_pc, fmt='%.6f')
