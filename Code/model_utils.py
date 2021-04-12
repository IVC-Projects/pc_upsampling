import os
import tensorflow as tf
from tf_ops.emd import tf_auctionmatch
from tf_ops.CD import tf_nndistance
from tf_ops.sampling import tf_sampling
from tf_ops.grouping.tf_grouping import query_ball_point, group_point, knn_point
from tf_ops.sampling.tf_sampling import gather_point, farthest_point_sample
import math

def pre_load_checkpoint(checkpoint_dir):
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        # print(" [*] Reading checkpoint from {}".format(ckpt.model_checkpoint_path))
        epoch_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        return epoch_step,ckpt.model_checkpoint_path
    else:
        return 0,None


def get_repulsion_loss4(pred, nsample=20, radius=0.07):
    # pred: (batch_size, npoint,3)
    idx, pts_cnt = query_ball_point(radius, nsample, pred, pred)
    tf.summary.histogram('smooth/unque_index', pts_cnt)

    grouped_pred = group_point(pred, idx)  # (batch_size, npoint, nsample, 3)
    grouped_pred -= tf.expand_dims(pred, 2)

    ##get the uniform loss
    h = 0.03
    dist_square = tf.reduce_sum(grouped_pred ** 2, axis=-1)
    dist_square, idx = tf.nn.top_k(-dist_square, 5)
    dist_square = -dist_square[:, :, 1:]  # remove the first one
    dist_square = tf.maximum(1e-12,dist_square)
    dist = tf.sqrt(dist_square)
    weight = tf.exp(-dist_square/h**2)
    uniform_loss = tf.reduce_mean(radius-dist*weight)
    return uniform_loss


def get_emd_loss(pred, gt, radius):
    """ pred: BxNxC,
        label: BxN, """
    batch_size = pred.get_shape()[0].value
    matchl_out, matchr_out = tf_auctionmatch.auction_match(pred, gt)
    matched_out = tf_sampling.gather_point(gt, matchl_out)
    dist = tf.reshape((pred - matched_out) ** 2, shape=(batch_size, -1))
    dist = tf.reduce_mean(dist, axis=1, keep_dims=True)
    dist_norm = dist / radius

    emd_loss = tf.reduce_mean(dist_norm)
    return emd_loss,matchl_out

def get_cd_loss(pred, gt, radius):
    """ pred: BxNxC,
        label: BxN, """
    dists_forward, _, dists_backward, _ = tf_nndistance.nn_distance(gt, pred)
    #dists_forward is for each element in gt, the cloest distance to this element
    CD_dist = 0.8*dists_forward + 0.2*dists_backward
    CD_dist = tf.reduce_mean(CD_dist, axis=1)
    CD_dist_norm = CD_dist/radius
    cd_loss = tf.reduce_mean(CD_dist_norm)
    return cd_loss,None

def get_uniform_loss(pcd, percentages=[0.002,0.004,0.006,0.008,0.010,0.012,0.015], radius=1.0):
    B, N, C = pcd.get_shape().as_list()
    npoint = int(N * 0.05)
    loss = []
    for p in percentages:
        nsample = int(N * p)
        r = math.sqrt(p * radius)
        disk_area = math.pi * (radius ** 2) * p / nsample
        new_xyz = gather_point(pcd, farthest_point_sample(npoint, pcd))  # (batch_size, npoint, 3)
        idx, pts_cnt = query_ball_point(r, nsample, pcd, new_xyz)  # (batch_size, npoint, nsample)
        expect_len = tf.sqrt(disk_area)  # using square
        grouped_pcd = group_point(pcd, idx)#(batch_size, npoint, nsample, channel)
        grouped_pcd = tf.concat(tf.unstack(grouped_pcd, axis=1), axis=0)
        var, _ = knn_point(2, grouped_pcd, grouped_pcd)
        uniform_dis = -var[:, :, 1:]
        uniform_dis = tf.sqrt(tf.abs(uniform_dis + 1e-8))
        uniform_dis = tf.reduce_mean(uniform_dis, axis=[-1])
        uniform_dis = tf.square(uniform_dis - expect_len) / (expect_len + 1e-8)
        uniform_dis = tf.reshape(uniform_dis, [-1])
        mean, variance = tf.nn.moments(uniform_dis, axes=0)  
        mean = mean * math.pow(p * 100, 2)
        loss.append(mean)
    return tf.add_n(loss) / len(percentages)


if __name__ == '__main__':
    gt = tf.constant([[[1,0,0],[2,0,0],[3,0,0],[4,0,0]]],tf.float32)
    pred = tf.constant([[[-10,0,0], [1,0, 0], [2,0, 0], [3,0,0]]],tf.float32)

    dists_forward, idx1, dists_backward, idx2 = tf_nndistance.nn_distance(gt, pred)
    with tf.Session() as sess:
        print(idx1.eval()) # for each element in gt, the idx of pred
        print(idx2.eval()) # for each element in pred,
