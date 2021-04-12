""" PointNet++ Layers

Author: Charles R. Qi
Date: November 2017
"""

import os
import sys
from tf_ops.sampling.tf_sampling import farthest_point_sample, gather_point
from tf_ops.grouping.tf_grouping import query_ball_point, group_point, knn_point
from tf_ops.interpolation.tf_interpolate import three_nn, three_interpolate
import tensorflow as tf
import numpy as np
import Utils.tf_util, Utils.tf_util2

def sample_and_group(npoint, radius, nsample, xyz, points, tnet_spec=None, knn=False, use_xyz=True):
    '''
    Input:
        npoint: int32
        radius: float32
        nsample: int32
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        tnet_spec: dict (keys: mlp, mlp2, is_training, bn_decay), if None do not apply tnet
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
    '''

    new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz)) # (batch_size, npoint, 3)
    if knn:
        _,idx = knn_point(nsample, xyz, new_xyz)
    else:
        if np.isscalar(radius):
            idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
        else:
            idx_list = []
            for radius_one, xyz_one, new_xyz_one in zip(tf.unstack(radius,axis=0), tf.unstack(xyz, axis=0),tf.unstack(new_xyz, axis=0)):
                idx_one, _ = query_ball_point(radius_one, nsample, tf.expand_dims(xyz_one, axis=0), tf.expand_dims(new_xyz_one, axis=0))
                idx_list.append(idx_one)
            idx = tf.stack(idx_list, axis=0)
            idx = tf.squeeze(idx, axis=1)

    grouped_xyz = group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
    grouped_xyz_for_point = grouped_xyz - tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization
    '''
    if tnet_spec is not None:
        grouped_xyz = tnet(grouped_xyz, tnet_spec)
    '''
    if points is not None:
        grouped_points = group_point(points, idx) # (batch_size, npoint, nsample, channel)
        if use_xyz:
            # new_points = tf.concat([grouped_xyz, tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]),grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)
            new_points = tf.concat([grouped_xyz_for_point, grouped_points],axis=-1)  # (batch_size, npoint, nample, 3+channel)
        else:
            new_points = grouped_points
    else:
        # new_points =  tf.concat([grouped_xyz, tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1])], axis=-1)
        new_points = grouped_xyz_for_point

    return new_xyz, new_points, idx, grouped_xyz


def sample_and_group_all(xyz, points, use_xyz=True):
    '''
    Inputs:
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Outputs:
        new_xyz: (batch_size, 1, 3) as (0,0,0)
        new_points: (batch_size, 1, ndataset, 3+channel) TF tensor
    Note:
        Equivalent to sample_and_group with npoint=1, radius=inf, use (0,0,0) as the centroid
    '''
    batch_size = xyz.get_shape()[0].value
    nsample = xyz.get_shape()[1].value
    new_xyz = tf.constant(np.tile(np.array([0,0,0]).reshape((1,1,3)), (batch_size,1,1)),dtype=tf.float32) # (batch_size, 1, 3)
    idx = tf.constant(np.tile(np.array(range(nsample)).reshape((1,1,nsample)), (batch_size,1,1)))
    grouped_xyz = tf.reshape(xyz, (batch_size, 1, nsample, 3)) # (batch_size, npoint=1, nsample, 3)
    if points is not None:
        if use_xyz:
            new_points = tf.concat([xyz, points], axis=2) # (batch_size, 16, 259)
        else:
            new_points = points
        new_points = tf.expand_dims(new_points, 1) # (batch_size, 1, 16, 259)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, idx, grouped_xyz


def pointnet_sa_module(xyz, points, npoint, radius, nsample, mlp, mlp2, is_training,
                       bn_decay, scope, bn=True, ibn=False, pooling='max', tnet_spec=None, knn=False, use_xyz=True, as_neighbor=8, weight_decay=None):
    ''' PointNet Set Abstraction (SA) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            batch_radius: the size of each object
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    with tf.variable_scope(scope) as sc:

        #new_points:(batch_size, npoint, nsample, 3+channel)
        _, new_points, idx, grouped_xyz = sample_and_group(npoint, radius, nsample, xyz, points, tnet_spec, knn, use_xyz)
        # (B, npoint, 1, 3),(B,npoint,nsample,C)
        new_xyz, new_points = AdaptiveSampling(grouped_xyz, new_points, as_neighbor, is_training, bn_decay,
                                                weight_decay, scope, bn)
        if mlp2 is None: mlp2 = []
        for i, num_out_channel in enumerate(mlp):
            new_points = Utils.tf_util2.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, ibn=ibn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay) 
        if pooling=='avg':
            new_points = tf.layers.average_pooling2d(new_points, [1,nsample], [1,1], padding='VALID', name='avgpool1')
        elif pooling=='weighted_avg':
            with tf.variable_scope('weighted_avg1'):
                dists = tf.norm(grouped_xyz,axis=-1,ord=2,keep_dims=True)
                exp_dists = tf.exp(-dists * 5)
                weights = exp_dists/tf.reduce_sum(exp_dists,axis=2,keep_dims=True) # (batch_size, npoint, nsample, 1)
                new_points *= weights # (batch_size, npoint, nsample, mlp[-1])
                new_points = tf.reduce_sum(new_points, axis=2, keep_dims=True)
        elif pooling=='max':
            new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True)
        elif pooling=='min':
            new_points = tf.layers.max_pooling2d(-1 * new_points, [1, nsample], [1, 1], padding='VALID',name='minpool1')
        elif pooling=='max_and_avg':
            avg_points = tf.layers.max_pooling2d(new_points, [1,nsample], [1,1], padding='VALID', name='maxpool1')
            max_points = tf.layers.average_pooling2d(new_points, [1,nsample],[1,1], padding='VALID', name='avgpool1')
            new_points = tf.concat([avg_points, max_points], axis=-1)
            
        if mlp2 is None: mlp2 = []
        for i, num_out_channel in enumerate(mlp2):
            new_points = Utils.tf_util2.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, ibn=ibn,is_training=is_training,
                                        scope='conv_post_%d'%(i), bn_decay=bn_decay) 
        new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlp2[-1])
        return new_xyz, new_points, idx

def pointnet_fp_module(xyz1, xyz2, points1, points2, mlp, is_training, bn_decay, scope, bn=True,ibn=False):
    ''' PointNet Feature Propogation (FP) Module
        Input:                                                                                                      
            xyz1: (batch_size, ndataset1, 3) TF tensor                                                              
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1                                           
            points1: (batch_size, ndataset1, nchannel1) TF tensor                                                   
            points2: (batch_size, ndataset2, nchannel2) TF tensor
            mlp: list of int32 -- output size for MLP on each point                                                 
        Return:
            new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0/dist),axis=2,keep_dims=True)
        norm = tf.tile(norm,[1,1,3])
        weight = (1.0/dist) / norm
        interpolated_points = three_interpolate(points2, idx, weight)

        if points1 is not None:
            new_points1 = tf.concat(axis=2, values=[interpolated_points, points1]) # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = interpolated_points
        new_points1 = tf.expand_dims(new_points1, 2)
        for i, num_out_channel in enumerate(mlp):
            new_points1 = Utils.tf_util2.conv2d(new_points1, num_out_channel, [1,1],
                                         padding='VALID', stride=[1,1],
                                         bn=bn, ibn=ibn,is_training=is_training,
                                         scope='conv_%d'%(i), bn_decay=bn_decay)
        new_points1 = tf.squeeze(new_points1, [2]) # B,ndataset1,mlp[-1]
        return new_points1

def AdaptiveSampling(group_xyz, group_feature, num_neighbor, is_training, bn_decay, weight_decay, scope, bn):
    with tf.variable_scope(scope) as sc:
        #（K,C） 每个FPS内的基于knn的搜索到的k个点和对应的特征值
        nsample, num_channel = group_feature.get_shape()[-2:]
        if num_neighbor == 0:
            new_xyz = group_xyz[:, :, 0, :]
            new_feature = group_feature[:, :, 0, :]
            return new_xyz, new_feature
        #取出每个采样点最近邻的num_neighbor个点（B,npoint,num_neighbor,3）
        shift_group_xyz = group_xyz[:, :, :num_neighbor, :]
        #同理，最近邻的num_neighbor的对应特征：（B,npoint,num_neighbor,C）
        shift_group_points = group_feature[:, :, :num_neighbor, :]

        #(B, npoint, k, 1 + num_channel)
        sample_weight = SampleWeights(shift_group_points, shift_group_xyz, [32, 1 + num_channel], is_training, bn_decay, weight_decay, scope, bn)
        '''以下的new_weight_xyz和new_weight_feture即邻居点集的特征、坐标的权重矩阵。'''
        # (B, npoint, k, 3):这里第三个维度k本身就代表权重值了(空间上想一下张量)，所以相当于减少了一个维度，如此一来就只有增加一个维度
        new_weight_xyz = tf.tile(tf.expand_dims(sample_weight[:,:,:, 0],axis=-1), [1, 1, 1, 3])
        #(B, npoint, k, num_channel)
        new_weight_feture = sample_weight[:,:,:, 1:]
        #（B,npoint,num_neighbor,3）*(B, npoint, k, 3)=(B, npoint, k, 3)->(B, npoint, 1, 3)
        new_xyz = tf.reduce_sum(tf.multiply(shift_group_xyz, new_weight_xyz), axis=[2])
        #同上：（B,npoint,1,C）
        # 这里做了改动，这样就把neighborhood ball中的点给矫正了，之前是在这一步上再进行了一个max得到了每个采样点的特征
        new_feature = tf.multiply(shift_group_points, new_weight_feture)
        #(B, npoint, 1, 3),（B,npoint,8,C）
        return new_xyz, new_feature

def SampleWeights(new_point, grouped_xyz, mlps, is_training, bn_decay, weight_decay, scope, bn=True, scaled=True):
    """Input
        grouped_feature: (batch_size, npoint, nsample, channel) TF tensor
        grouped_xyz: (batch_size, npoint, nsample, 3) : (B,npoint,k,3）
        new_point: (batch_size, npoint, nsample, channel) :（B,npoint,k,C）
        Output
        (batch_size, npoint, nsample, 1)
    """
    with tf.variable_scope(scope) as sc:
        #B,npoint,k,C
        batch_size, npoint, nsample, channel = new_point.get_shape()
        bottleneck_channel = max(32,channel//2)#channel//2求模计算，/：求浮点数值，//：求整数值，即求模
        #每个采样点对应的k个最近邻点减去第一个最近邻点(不知道这第一个是自己还是采样点的第一个最近邻点)得到相对坐标
        normalized_xyz = grouped_xyz - tf.tile(tf.expand_dims(grouped_xyz[:, :, 0, :], 2), [1, 1, nsample, 1])
        #得到每个邻居点的新特征：（B,npoint,k,C+3）
        new_point = tf.concat([normalized_xyz, new_point], axis=-1) # (batch_size, npoint, nsample, channel+3)

        #key矩阵
        transformed_feature = Utils.tf_util.conv2d(new_point, bottleneck_channel * 2, [1, 1],
                                             padding='VALID', stride=[1, 1],
                                             bn=bn, is_training=is_training,
                                             scope='conv_kv_ds', bn_decay=bn_decay, weight_decay=weight_decay,
                                             activation_fn=None)

        #query矩阵
        transformed_new_point = Utils.tf_util.conv2d(new_point, bottleneck_channel, [1, 1],
                                               padding='VALID', stride=[1, 1],
                                               bn=bn, is_training=is_training,
                                               scope='conv_query_ds', bn_decay=bn_decay, weight_decay=weight_decay,
                                               activation_fn=None)
        #key矩阵：(B,npoint,k,bottleneck_channel)
        transformed_feature1 = transformed_feature[:, :, :, :bottleneck_channel]
        #value矩阵：(B,npoint,k,bottleneck_channel)
        feature = transformed_feature[:, :, :, bottleneck_channel:]
        #attention矩阵：(B,npoint,k,k)
        weights = tf.matmul(transformed_new_point, transformed_feature1, transpose_b=True)  # (batch_size, npoint, nsample, nsample)
        if scaled:
            weights = weights / tf.sqrt(tf.cast(bottleneck_channel, tf.float32))
        #权重矩阵（B,npoint,k,k），最后一个维度代表每个点和其他k个点的softmax值，可以理解为关联程度，即权重值
        #这样就得到了采样点的k近邻的一个关联权重矩阵
        weights = tf.nn.softmax(weights, axis=-1)
        channel = bottleneck_channel

        #（B,npoint,k,k）*（B,npoint,k,bottleneck_channel）=（B,npoint,k,bottleneck_channel）
        new_group_features = tf.matmul(weights, feature)
        #（B,npoint,k,bottleneck_channel）：得到所有邻居点新的feature，即增强邻居点feature
        new_group_features = tf.reshape(new_group_features, (batch_size, npoint, nsample, channel))
        for i, c in enumerate(mlps):
            activation = tf.nn.relu if i < len(mlps) - 1 else None
            new_group_features = Utils.tf_util.conv2d(new_group_features, c, [1, 1],
                                               padding='VALID', stride=[1, 1],
                                               bn=bn, is_training=is_training,
                                               scope='mlp2_%d' % (i), bn_decay=bn_decay, weight_decay=weight_decay,
                                               activation_fn=activation)
        new_group_weights = tf.nn.softmax(new_group_features, axis=2)  # (batch_size, npoint,nsample, mlp[-1])
        #softmax:计算出每个邻居点的对应特征channel的权重矩阵，即每个channel中的关联重要程度
        #(B, npoint, k, 1 + num_channel)
        return new_group_weights