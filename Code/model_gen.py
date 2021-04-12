import tensorflow as tf
from Utils import tf_util2
from Utils.tf_util2 import feature_extraction

def placeholder_inputs(batch_size, num_point,up_ratio = 4):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    pointclouds_gt = tf.placeholder(tf.float32, shape=(batch_size, num_point*up_ratio, 3))
    pointclouds_normal = tf.placeholder(tf.float32, shape=(batch_size, num_point * up_ratio, 3))
    pointclouds_radius = tf.placeholder(tf.float32, shape=(batch_size))
    return pointclouds_pl, pointclouds_gt,pointclouds_normal, pointclouds_radius


def get_gen_model(point_cloud, is_training, scope, reuse=None, use_rv=False, use_bn = False, use_ibn = False, bn_decay=None, up_ratio = 4):
    l0_xyz = point_cloud[:, :, 0:3]
    with tf.variable_scope(scope,reuse=reuse) as sc:
        batch_size = point_cloud.get_shape()[0].value
        num_point = point_cloud.get_shape()[1].value
        #(B, N, 1, C)
        concat_feat = feature_extraction(l0_xyz, scope='feature_extraction', is_training=is_training, bn_decay=None)

        with tf.variable_scope('up_layer', reuse=reuse):

                new_grid = []
                for i in range(up_ratio):
                    concat_feat_res = concat_feat
                    concat_feat_res = tf_util2.conv2d(concat_feat_res, 256, [1, 1],
                                                  padding='VALID', stride=[1, 1],
                                                  bn=False, is_training=is_training,
                                                  scope='reslayer1_%d' % (i), bn_decay=bn_decay)

                    concat_feat_res = tf_util2.conv2d(concat_feat_res, 64, [1, 1],
                                                      padding='VALID', stride=[1, 1],
                                                      bn=use_bn, is_training=is_training,
                                                      scope='reslayer2_%d' % (i), bn_decay=bn_decay)

                    concat_feat_res = tf_util2.conv2d(concat_feat_res, 2, [1, 1],
                                                      padding='VALID', stride=[1, 1],
                                                      bn=use_bn, is_training=is_training,
                                                      scope='reslayer3_%d' % (i), bn_decay=bn_decay)
                    new_grid.append(concat_feat_res)
                new_grids = tf.concat(new_grid, axis=1)

                net_tile = tf.tile(concat_feat, [1, up_ratio, 1, 1]) #(B,rN,1,C)
                net_tile_concat = tf.concat([net_tile, new_grids], axis=-1) #(B,rN,1,C+2)

                # (B,rN,1,C)
                net_tile_concat = tf_util2.conv2d(net_tile_concat, 256, [1, 1],
                                                  padding='VALID', stride=[1, 1],
                                                  bn=False, is_training=is_training,
                                                  scope='fold_layer0', bn_decay=bn_decay)

                net_tile_concat = tf_util2.conv2d(net_tile_concat, 64, [1, 1],
                                                  padding='VALID', stride=[1, 1],
                                                  bn=False, is_training=is_training,
                                                  scope='fold_layer1', bn_decay=bn_decay)

                net_tile_concat = tf_util2.conv2d(net_tile_concat, 3, [1, 1],
                                                  padding='VALID', stride=[1, 1],
                                                  bn=False, is_training=is_training,
                                                  scope='fold_layer2', bn_decay=bn_decay)
                

                #(B,rN,1,C+3)
                net_tile_concat2 = tf.concat([net_tile, net_tile_concat], axis=-1)

                net_tile_concat2 = tf_util2.conv2d(net_tile_concat2, 256, [1, 1],
                                                   padding='VALID', stride=[1, 1],
                                                   bn=False, is_training=is_training,
                                                   scope='fold_layer3', bn_decay=bn_decay)

                net_tile_concat2 = tf_util2.conv2d(net_tile_concat2, 64, [1, 1],
                                                   padding='VALID', stride=[1, 1],
                                                   bn=False, is_training=is_training,
                                                   scope='fold_layer4', bn_decay=bn_decay)

                coord = tf_util2.conv2d(net_tile_concat2, 3, [1, 1],
                                        padding='VALID', stride=[1, 1],
                                        bn=False, is_training=is_training,
                                        scope='fold_layer5', bn_decay=bn_decay,
                                        activation_fn=None, weight_decay=0.0) #(B,rN,1,3)
        coord = tf.squeeze(coord, [2])
    return coord,None