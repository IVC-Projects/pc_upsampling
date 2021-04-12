import tensorflow as tf
import numpy as np

def lrelu(x, alpha=0.2):
  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


# def lrelu2(x, leak=0.2, name="lrelu"):
#     with tf.variable_scope(name):
#         f1 = 0.5 * (1 + leak)
#         f2 = 0.5 * (1 - leak)
#         return f1 * x + f2 * abs(x)

def instance_norm(net, train=True,weight_decay=0.00001):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(net, [1, 2], keep_dims=True)

    shift = tf.get_variable('shift',shape=var_shape,
                            initializer=tf.zeros_initializer,
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
    scale = tf.get_variable('scale', shape=var_shape,
                            initializer=tf.ones_initializer,
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
    epsilon = 1e-3
    normalized = (net - mu) / tf.square(sigma_sq + epsilon)
    return scale * normalized + shift

def conv2d(inputs,
           num_output_channels,
           kernel_size,
           scope=None,
           stride=[1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.00001,
           activation_fn=tf.nn.relu,
           bn=False,
           ibn = False,
           bn_decay=None,
           use_bias = True,
           is_training=None,
           reuse=None):
  """ 2D convolution with non-linear operation.

  Args:
    inputs: 4-D tensor variable BxHxWxC
    num_output_channels: int
    kernel_size: a list of 2 ints
    scope: string
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope,reuse=reuse) as sc:
      if use_xavier:
          initializer = tf.contrib.layers.xavier_initializer()
      else:
          initializer = tf.truncated_normal_initializer(stddev=stddev)

      outputs = tf.layers.conv2d(inputs,num_output_channels,kernel_size,stride,padding,
                                 kernel_initializer=initializer,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                 bias_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                 use_bias=use_bias,reuse=None)
      assert not (bn and ibn)
      if bn:
          outputs = tf.layers.batch_normalization(outputs,momentum=bn_decay,training=is_training,renorm=False,fused=True)
          #outputs = tf.contrib.layers.batch_norm(outputs,is_training=is_training)
      if ibn:
          outputs = instance_norm(outputs,is_training)


      if activation_fn is not None:
        outputs = activation_fn(outputs)

      return outputs


def fully_connected(inputs,
                    num_outputs,
                    scope,
                    use_xavier=True,
                    stddev=1e-3,
                    weight_decay=0.00001,
                    activation_fn=tf.nn.relu,
                    bn=False,
                    bn_decay=None,
                    use_bias = True,
                    is_training=None):
    """ Fully connected layer with non-linear operation.

    Args:
      inputs: 2-D tensor BxN
      num_outputs: int

    Returns:
      Variable tensor of size B x num_outputs.
    """

    with tf.variable_scope(scope) as sc:
        if use_xavier:
            initializer = tf.contrib.layers.xavier_initializer()
        else:
            initializer = tf.truncated_normal_initializer(stddev=stddev)

        outputs = tf.layers.dense(inputs,num_outputs,
                                  use_bias=use_bias,kernel_initializer=initializer,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                  bias_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                  reuse=None)

        if bn:
            outputs = tf.layers.batch_normalization(outputs, momentum=bn_decay, training=is_training, renorm=False)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs


def pairwise_distance(point_cloud):
  """Compute pairwise distance of a point cloud.
  Args:
    point_cloud: tensor (batch_size, num_points, num_dims):(B,N,3)
  Returns:
    pairwise distance: (batch_size, num_points, num_points)
  """
  og_batch_size = point_cloud.get_shape().as_list()[0]
  point_cloud = tf.squeeze(point_cloud)
  if og_batch_size == 1:
    point_cloud = tf.expand_dims(point_cloud, 0)

  point_cloud_transpose = tf.transpose(point_cloud, perm=[0, 2, 1])#(B,3,N)
  point_cloud_inner = tf.matmul(point_cloud, point_cloud_transpose)
  point_cloud_inner = -2*point_cloud_inner
  point_cloud_square = tf.reduce_sum(tf.square(point_cloud), axis=-1, keepdims=True)#(B,N,1)
  point_cloud_square_tranpose = tf.transpose(point_cloud_square, perm=[0, 2, 1])#(B,1,N)
  return point_cloud_square + point_cloud_inner + point_cloud_square_tranpose#(B,N,N)

def knn(adj_matrix, k=20):
  """Get KNN based on the pairwise distance.
  Args:
    pairwise distance: (batch_size, num_points, num_points)
    k: int
  Returns:
    nearest neighbors: (batch_size, num_points, k)
    Returns:
    values: The `k` largest elements along each last dimensional slice.
    indices: The indices of `values` within the last dimension of `input`.
  """
  neg_adj = -adj_matrix
  _, nn_idx = tf.nn.top_k(neg_adj, k=k)
  return nn_idx

def get_edge_feature(point_cloud, nn_idx, k=20):
  """Construct edge feature for each point
  Args:
    point_cloud: (batch_size, num_points, 1, num_dims)
    nn_idx: (batch_size, num_points, k)
    k: int

  Returns:
    edge features: (batch_size, num_points, k, num_dims)
  """
  og_batch_size = point_cloud.get_shape().as_list()[0]#B
  point_cloud = tf.squeeze(point_cloud)#(B,N,3)
  if og_batch_size == 1:
    point_cloud = tf.expand_dims(point_cloud, 0)

  point_cloud_central = point_cloud#(B,N,3)

  point_cloud_shape = point_cloud.get_shape()
  batch_size = point_cloud_shape[0].value#B
  num_points = point_cloud_shape[1].value#N
  num_dims = point_cloud_shape[2].value#3

  idx_ = tf.range(batch_size) * num_points#创建一个序列，序列大小B×N
  idx_ = tf.reshape(idx_, [batch_size, 1, 1]) #这里将上面的给reshape得到(B×N,1,1)

  point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])#(B×N,3)
  point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx+idx_)#根据索引的结果给flat插入对应的索引值(B,N,k,3)，即每个点对应的k个
  point_cloud_central = tf.expand_dims(point_cloud_central, axis=-2)#(B,N,1,3)

  #其实这里k最近邻，其中一个最近的肯定是自己，也就是有个点的pair distance是0
  point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])#(B,N,k,3)对每个点复制k份，用于计算每个点的最近k个点和它的相对距离

  #(420, 2048, 20, 6):(B,N,k,该点坐标+k点集和该点的相对距离值坐标)
  edge_feature = tf.concat([point_cloud_central, point_cloud_neighbors-point_cloud_central], axis=-1)
  #(B, N, k, 6)
  return edge_feature

def gen_grid(up_ratio):
    import math
    """
    output [num_grid_point, 2]
    """
    sqrted = int(math.sqrt(up_ratio))+1
    for i in range(1,sqrted+1).__reversed__():
        if (up_ratio%i) == 0:
            num_x = i
            num_y = up_ratio//i
            break
    grid_x = tf.lin_space(-0.2, 0.2, num_x)
    grid_y = tf.lin_space(-0.2, 0.2, num_y)

    x, y = tf.meshgrid(grid_x, grid_y)
    grid = tf.reshape(tf.stack([x, y], axis=-1), [-1, 2])  # [2, 2, 2] -> [4, 2]
    return grid


def gen_grid2(batch_size, num_point, up_ratio):
    disturbance_list = []
    import math
    sqrted = int(math.sqrt(up_ratio)) + 1
    for i in range(1, sqrted + 1).__reversed__():
        if (up_ratio % i) == 0:
            num_x = i
            num_y = up_ratio // i
            break
    grid_x = tf.lin_space(-0.2, 0.2, num_x)
    grid_y = tf.lin_space(-0.2, 0.2, num_y)
    x, y = tf.meshgrid(grid_x, grid_y)
    grid = tf.reshape(tf.stack([x, y], axis=-1), [-1, 2])#(4, 2)

    with tf.Session() as sess:
        disturbance = tf.convert_to_tensor(grid, dtype='float32')
        for i in range(up_ratio):
            disturbance_one= tf.expand_dims(tf.expand_dims(tf.expand_dims(disturbance[i,], 0), 0), 0)#(1,2)->(1,1,1,2)
            disturbance_one = tf.tile(disturbance_one, [1, num_point, 1, 1])#(1,N,1,2)
            disturbance_one = tf.tile(disturbance_one, [batch_size, 1, 1, 1])#(B,N,1,2)
            disturbance_list.append(disturbance_one)
    disturbance_feature = tf.concat(disturbance_list, axis = 1)#(B,4N,1,2)
    return disturbance_feature

def feature_extraction(inputs, scope='feature_extraction2', is_training=True, bn_decay=None):
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        use_bn = False
        use_ibn = False
        growth_rate = 32
        knn = 16
        comp = growth_rate*2
        l0_features = tf.expand_dims(inputs, axis=2)
        l0_features = conv2d(l0_features, 64, [1, 1],
                                     padding='VALID', scope='layer0', is_training=is_training, bn=use_bn, ibn=use_ibn,
                                     bn_decay=bn_decay, activation_fn=None)
        l0_features = tf.squeeze(l0_features, axis=2)
        # encoding layer
        l1_features, l1_idx = dense_conv(l0_features, growth_rate=growth_rate, k=knn,
                                                  scope="layer1", scope_num=0, is_training=is_training, bn=use_bn, bn_decay=bn_decay)
        l1_features = tf.concat([l1_features, l0_features], axis=-1)  # ((128+64)+64)+64

        l2_features = conv1d(l1_features, comp, 1,
                                     padding='VALID', scope='layer2_prep', is_training=is_training, bn=use_bn, ibn=use_ibn,
                                     bn_decay=bn_decay)
        l2_features, l2_idx = dense_conv(l2_features, growth_rate=growth_rate, k=knn,
                                                  scope="layer2", scope_num=1, is_training=is_training, bn=use_bn, bn_decay=bn_decay)
        l2_features = tf.concat([l2_features, l1_features], axis=-1)  # ((256+128)+128)+(((128+64)+64)+64)
        l3_features = conv1d(l2_features, comp*2, 1,
                                    padding='VALID', scope='layer3_prep', is_training=is_training, bn=use_bn, ibn=use_ibn,
                                    bn_decay=bn_decay)
        l3_features = tf.expand_dims(l3_features, axis=2)
    return l3_features

def conv1d(inputs,
           num_output_channels,
           kernel_size,
           scope=None,
           stride=1,
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.00001,
           activation_fn=tf.nn.relu,
           bn=False,
           ibn=False,
           bn_decay=None,
           use_bias=True,
           is_training=None,
           reuse=None):
    """ 1D convolution with non-linear operation.

    Args:
        inputs: 3-D tensor variable BxHxWxC
        num_output_channels: int
        kernel_size: int
        scope: string
        stride: a list of 2 ints
        padding: 'SAME' or 'VALID'
        use_xavier: bool, use xavier_initializer if true
        stddev: float, stddev for truncated_normal init
        weight_decay: float
        activation_fn: function
        bn: bool, whether to use batch norm
        bn_decay: float or float tensor variable in [0,1]
        is_training: bool Tensor variable

    Returns:
        Variable tensor
    """
    with tf.variable_scope(scope, reuse=reuse):
        if use_xavier:
            initializer = tf.contrib.layers.xavier_initializer()
        else:
            initializer = tf.truncated_normal_initializer(stddev=stddev)

        outputs = tf.layers.conv1d(inputs, num_output_channels, kernel_size, stride, padding,
                                   kernel_initializer=initializer,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                       weight_decay),
                                   bias_regularizer=tf.contrib.layers.l2_regularizer(
                                       weight_decay),
                                   use_bias=use_bias, reuse=None)
        assert not (bn and ibn)
        if bn:
            outputs = tf.layers.batch_normalization(
                outputs, momentum=bn_decay, training=is_training, renorm=False, fused=True)
            # outputs = tf.contrib.layers.batch_norm(outputs,is_training=is_training)
        if ibn:
            outputs = instance_norm(outputs, is_training)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs



def conv2d(inputs,
           num_output_channels,
           kernel_size,
           scope=None,
           stride=[1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.00001,
           activation_fn=tf.nn.relu,
           bn=False,
           ibn = False,
           bn_decay=None,
           use_bias = True,
           is_training=None,
           reuse=tf.AUTO_REUSE):
  """ 2D convolution with non-linear operation.

  Args:
    inputs: 4-D tensor variable BxHxWxC
    num_output_channels: int
    kernel_size: a list of 2 ints
    scope: string
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope,reuse=reuse) as sc:
      if use_xavier:
          initializer = tf.contrib.layers.xavier_initializer()
      else:
          initializer = tf.truncated_normal_initializer(stddev=stddev)

      outputs = tf.layers.conv2d(inputs,num_output_channels,kernel_size,stride,padding,
                                 kernel_initializer=initializer,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                 bias_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                 use_bias=use_bias,reuse=None)
      assert not (bn and ibn)
      if bn:
          outputs = tf.layers.batch_normalization(outputs,momentum=bn_decay,training=is_training,renorm=False,fused=True)
          #outputs = tf.contrib.layers.batch_norm(outputs,is_training=is_training)
      if ibn:
          outputs = instance_norm(outputs,is_training)


      if activation_fn is not None:
        outputs = activation_fn(outputs)

      return outputs

def get_edge_feature(point_cloud, k=16, idx=None):
    """Construct edge feature for each point
    Args:
        point_cloud: (batch_size, num_points, 1, num_dims)
        nn_idx: (batch_size, num_points, k, 2)
        k: int
    Returns:
        edge features: (batch_size, num_points, k, num_dims)
    """
    if idx is None:
        _, idx = knn_point_2(k + 1, point_cloud, point_cloud, unique=True, sort=True)
        idx = idx[:, :, 1:, :]

    # [N, P, K, Dim]
    point_cloud_neighbors = tf.gather_nd(point_cloud, idx)
    point_cloud_central = tf.expand_dims(point_cloud, axis=-2)

    point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])

    edge_feature = tf.concat(
        [point_cloud_central, point_cloud_neighbors - point_cloud_central], axis=-1)
    return edge_feature, idx

'''
def dense_conv(feature, n=3, growth_rate=64, k=16, scope='dense_conv', **kwargs):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        y, idx = get_edge_feature(feature, k=k, idx=None)  # [B N K 2*C]
        for i in range(n):
            if i == 0:
                y = tf.concat([
                    conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l%d' % i, **kwargs),
                    tf.tile(tf.expand_dims(feature, axis=2), [1, 1, k, 1])], axis=-1)

            elif i == n - 1:
                y = tf.concat([
                    conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l%d' % i, activation_fn=None, **kwargs),
                    y], axis=-1)
            else:
                y = tf.concat([
                    conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l%d' % i, **kwargs),
                    y], axis=-1)
        y = tf.reduce_max(y, axis=-2)
        return y, idx
'''

def dense_conv(feature, growth_rate=32, k=16, scope='dense_conv', scope_num= 0,  **kwargs):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        y, idx = get_edge_feature(feature, k=k, idx=None)  # [B N K 2*C]
        y = tf.concat([
            conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l_1_%d' % scope_num, **kwargs),
            tf.tile(tf.expand_dims(feature, axis=2), [1, 1, k, 1])], axis=-1)
        y = tf.concat([
            conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l_2_%d' % scope_num, activation_fn=None, **kwargs), y], axis=-1)
        y = tf.reduce_max(y, axis=-2)
        return y, idx

# return shape is (N, P, K, 2)
def knn_point_2(k, points, queries, sort=True, unique=True):
    """
    points: dataset points (N, P0, K)
    queries: query points (N, P, K)
    return indices is (N, P, K, 2) used for tf.gather_nd(points, indices)
    distances (N, P, K)
    """
    #import pdb
    #pdb.set_trace()
    with tf.name_scope("knn_point"):
        batch_size = tf.shape(queries)[0]
        point_num = tf.shape(queries)[1]

        D = batch_distance_matrix_general(queries, points)
        if unique:
            prepare_for_unique_top_k(D, points)
        distances, point_indices = tf.nn.top_k(-D, k=k, sorted=sort)  # (N, P, K)
        batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, point_num, k, 1))
        indices = tf.concat([batch_indices, tf.expand_dims(point_indices, axis=3)], axis=3)
        return -distances, indices

# A shape is (N, P_A, C), B shape is (N, P_B, C)
# D shape is (N, P_A, P_B)
def batch_distance_matrix_general(A, B):
    r_A = tf.reduce_sum(A * A, axis=2, keepdims=True)
    r_B = tf.reduce_sum(B * B, axis=2, keepdims=True)
    m = tf.matmul(A, tf.transpose(B, perm=(0, 2, 1)))
    D = r_A - 2 * m + tf.transpose(r_B, perm=(0, 2, 1))
    return D


# add a big value to duplicate columns
def prepare_for_unique_top_k(D, A):
    indices_duplicated = tf.py_func(find_duplicate_columns, [A], tf.int32)
    D += tf.reduce_max(D)*tf.cast(indices_duplicated, tf.float32)

# A shape is (N, P, C)
def find_duplicate_columns(A):
    N = A.shape[0]
    P = A.shape[1]
    indices_duplicated = np.ones((N, 1, P), dtype=np.int32)
    for idx in range(N):
        _, indices = np.unique(A[idx], return_index=True, axis=0)
        indices_duplicated[idx, :, indices] = 0
    return indices_duplicated

def attention_unit(inputs, scope='attention_unit',is_training=True, scope_num=0):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        dim = inputs.get_shape()[-1].value
        layer = dim//4
        f = conv2d(inputs, layer, [1, 1],
                              padding='VALID', stride=[1, 1],
                              bn=False, is_training=is_training,
                              scope='conv_f_%d' % (scope_num), bn_decay=None)

        g = conv2d(inputs, layer, [1, 1],
                            padding='VALID', stride=[1, 1],
                            bn=False, is_training=is_training,
                            scope='conv_g_%d' % (scope_num), bn_decay=None)

        h = conv2d(inputs, dim, [1, 1],
                            padding='VALID', stride=[1, 1],
                            bn=False, is_training=is_training,
                            scope='conv_h_%d' % (scope_num), bn_decay=None)


        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta = tf.nn.softmax(s, axis=-1)  # attention map

        o = tf.matmul(beta, hw_flatten(h))   # [bs, N, N]*[bs, N, c]->[bs, N, c]
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

        o = tf.reshape(o, shape=inputs.shape)  # [bs, h, w, C]
        x = gamma * o + inputs

    return x

def flatten(input):
    return tf.reshape(input, [-1, np.prod(input.get_shape().as_list()[1:])])

def hw_flatten(x):
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])