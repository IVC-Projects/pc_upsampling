# -*- coding: utf-8 -*-
# @Time        : 16/1/2019 5:49 PM
# @Description :
# @Author      : li rui hui
# @Email       : ruihuili@gmail.com
import tensorflow as tf
from tf_ops.sampling.tf_sampling import gather_point, farthest_point_sample
import Utils.tf_util2 as tf_util2
from Utils.tf_util2 import feature_extraction
class Generator(object):
    def __init__(self, opts,is_training, name="Generator"):
        self.opts = opts
        self.is_training = is_training
        self.name = name
        self.reuse = False
        self.num_point = self.opts.patch_num_point
        self.up_ratio = self.opts.up_ratio
        self.up_ratio_real = self.up_ratio + self.opts.more_up
        self.out_num_point = int(self.num_point*self.up_ratio)

    def __call__(self, inputs):
        with tf.variable_scope(self.name, reuse=self.reuse):
            concat_feat = feature_extraction(inputs, scope='feature_extraction', is_training=self.is_training, bn_decay=None)
            with tf.variable_scope('up_layer', reuse=self.reuse):
                new_grid = []
                for i in range(self.up_ratio):
                    concat_feat_res = concat_feat
                    concat_feat_res = tf_util2.conv2d(concat_feat_res, 256, [1, 1],
                                                      padding='VALID', stride=[1, 1],
                                                      bn=False, is_training=self.is_training,
                                                      scope='reslayer1_%d' % (i), bn_decay=None)

                    concat_feat_res = tf_util2.conv2d(concat_feat_res, 64, [1, 1],
                                                      padding='VALID', stride=[1, 1],
                                                      bn=False, is_training=self.is_training,
                                                      scope='reslayer2_%d' % (i), bn_decay=None)

                    concat_feat_res = tf_util2.conv2d(concat_feat_res, 2, [1, 1],
                                                      padding='VALID', stride=[1, 1],
                                                      bn=False, is_training=self.is_training,
                                                      scope='reslayer3_%d' % (i), bn_decay=None)
                    new_grid.append(concat_feat_res)
                new_grids = tf.concat(new_grid, axis=1)

                net_tile = tf.tile(concat_feat, [1, self.up_ratio, 1, 1])  # (B,rN,1,C)
                net_tile_concat = tf.concat([net_tile, new_grids], axis=-1)  # (B,rN,1,C+2)

                # (B,rN,1,C)
                net_tile_concat = tf_util2.conv2d(net_tile_concat, 256, [1, 1],
                                                  padding='VALID', stride=[1, 1],
                                                  bn=False, is_training=self.is_training,
                                                  scope='fold_layer0', bn_decay=None)

                net_tile_concat = tf_util2.conv2d(net_tile_concat, 64, [1, 1],
                                                  padding='VALID', stride=[1, 1],
                                                  bn=False, is_training=self.is_training,
                                                  scope='fold_layer1', bn_decay=None)

                net_tile_concat = tf_util2.conv2d(net_tile_concat, 3, [1, 1],
                                                  padding='VALID', stride=[1, 1],
                                                  bn=False, is_training=self.is_training,
                                                  scope='fold_layer2', bn_decay=None)

                # (B,rN,1,C+3)
                net_tile_concat2 = tf.concat([net_tile, net_tile_concat], axis=-1)

                net_tile_concat2 = tf_util2.conv2d(net_tile_concat2, 256, [1, 1],
                                                   padding='VALID', stride=[1, 1],
                                                   bn=False, is_training=self.is_training,
                                                   scope='fold_layer3', bn_decay=None)

                net_tile_concat2 = tf_util2.conv2d(net_tile_concat2, 64, [1, 1],
                                                   padding='VALID', stride=[1, 1],
                                                   bn=False, is_training=self.is_training,
                                                   scope='fold_layer4', bn_decay=None)

                coord = tf_util2.conv2d(net_tile_concat2, 3, [1, 1],
                                        padding='VALID', stride=[1, 1],
                                        bn=False, is_training=self.is_training,
                                        scope='fold_layer5', bn_decay=None,
                                        activation_fn=None, weight_decay=0.0)  # (B,rN,1,3)
            outputs = tf.squeeze(coord, [2])

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
        return outputs