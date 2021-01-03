from __future__ import print_function
import os,sys,argparse,platform
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
if platform.system()=="Windows":
    caffe_root = 'D:/CNN/caffe'
else:
    caffe_root = '~/CNN/caffe/'
sys.path.insert(0, caffe_root + '/python')
import caffe

@slim.add_arg_scope
def prelu(inputs, data_format='NHWC', scope=None):

    with tf.variable_scope(scope, default_name='prelu'):

        channel_dim = 1 if data_format == 'NCHW' else 3
        inputs_shape = inputs.get_shape().as_list()
        alpha_shape = [1 for i in range(len(inputs_shape))]
        alpha_shape[channel_dim] = inputs_shape[channel_dim]
        alpha = slim.model_variable(
            'weights', alpha_shape,
            initializer=tf.constant_initializer(0.25))

        outputs = tf.where(inputs > 0, inputs, inputs * alpha)

        return outputs

def fc2conv(var, v, *args):
    w, h, in_, out_ = var.shape.as_list()
    assert out_ == v.shape[0]
    v = v.reshape(out_, in_, w, h)
    # transpose wh to hw
    v = v.transpose(3, 2, 1, 0)
    
    # make the reg format from [x1, y1, x2, y2] to [y1, x1, y2, x2]
    if out_ == 4:
        v_ = [v[..., 1], v[..., 0], v[..., 3], v[..., 2]]
        v = np.stack(v_, axis=3)
        print(var.name, v.shape)
    
    # make the landmark output format from
    # [x1, x2, x3, x4, x5, y1, y2, y3, y4, y5] to
    # [y1, y2, y3, y4, y5, x1, x2, x3, x4, x5]
    if 'onet/conv6-3' in var.name:
        new_v = [v[..., 5:], v[..., :5]]
        new_v = np.concatenate(new_v, axis=3)
        print(var.name, v.shape)
        return new_v
    return v

def conv_t(var, v, *args):
    w, h, in_, out_ = var.shape.as_list()
    # transpose wh to hw
    v = v.transpose(3, 2, 1, 0)

    # change input image from rgb to bgr
    if 'conv1' in var.name:
        v = v[:, :, ::-1, :]
    
    # make the reg format from [x1, y1, x2, y2] to [y1, x1, y2, x2]
    if out_ == 4:
        v_ = [v[..., 1], v[..., 0], v[..., 3], v[..., 2]]
        v = np.stack(v_, axis=3)
        print(var.name, v.shape)
    return v

def conv_b_t(var, v, *args):
    out_ = var.shape.as_list()[0]

    # make the reg format from [x1, y1, x2, y2] to [y1, x1, y2, x2]
    if out_ == 4:
        v_ = [v[1], v[0], v[3], v[2]]
        v = np.asarray(v_)
        print(var.name, v.shape)
    
    # make the landmark output format from
    # [x1, x2, x3, x4, x5, y1, y2, y3, y4, y5] to
    # [y1, y2, y3, y4, y5, x1, x2, x3, x4, x5]
    if 'onet/conv6-3' in var.name:
        new_v = [v[5:], v[:5]]
        new_v = np.concatenate(new_v, axis=0)
        print(var.name, v.shape)
        v = new_v
    return v


def det1(images, data_format):
    net = slim.conv2d(images, 10, 3, stride=1, padding='VALID', scope='conv1')
    net = prelu(net, scope='PReLU1')
    net = slim.max_pool2d(net, 2, stride=2, padding='VALID', scope='pool1')

    net = slim.conv2d(net, 16, 3, stride=1, padding='VALID', scope='conv2')
    net = prelu(net, scope='PReLU2')

    net = slim.conv2d(net, 32, 3, stride=1, padding='VALID', scope='conv3')
    net = prelu(net, scope='PReLU3')

    prob = slim.conv2d(net, 2, 1, stride=1, padding='VALID', scope='conv4-1')
    prob = tf.nn.softmax(prob, axis=3 if data_format == 'NHWC' else 1, name='prob1')

    regress = slim.conv2d(net, 4, 1, stride=1, padding='VALID', scope='conv4-2')
    return prob, regress


def det2(images):
    net = slim.conv2d(images, 28, 3, stride=1, padding='VALID', scope='conv1')
    net = prelu(net, scope='prelu1')
    net = slim.max_pool2d(net, 3, stride=2, padding='SAME', scope='pool1')

    net = slim.conv2d(net, 48, 3, stride=1, padding='VALID', scope='conv2')
    net = prelu(net, scope='prelu2')
    net = slim.max_pool2d(net, 3, stride=2, padding='VALID', scope='pool2')

    net = slim.conv2d(net, 64, 2, stride=1, padding='VALID', scope='conv3')
    net = prelu(net, scope='prelu3')

    net = slim.conv2d(net, 128, 3, stride=1, padding='VALID', scope='conv4')
    net = prelu(net, scope='prelu4')

    prob = slim.conv2d(net, 2, 1, stride=1, padding='VALID', scope='conv5-1')
    prob = slim.flatten(prob)
    prob = tf.nn.softmax(prob, axis=1, name='prob1')

    regress = slim.conv2d(net, 4, 1, stride=1, padding='VALID', scope='conv5-2')
    regress = slim.flatten(regress)
    return prob, regress


def det3(images):
    net = slim.conv2d(images, 32, 3, stride=1, padding='VALID', scope='conv1')
    net = prelu(net, scope='prelu1')
    net = slim.max_pool2d(net, 3, stride=2, padding='SAME', scope='pool1')

    net = slim.conv2d(net, 64, 3, stride=1, padding='VALID', scope='conv2')
    net = prelu(net, scope='prelu2')
    net = slim.max_pool2d(net, 3, stride=2, padding='VALID', scope='pool2')

    net = slim.conv2d(net, 64, 3, stride=1, padding='VALID', scope='conv3')
    net = prelu(net, scope='prelu3')
    net = slim.max_pool2d(net, 2, stride=2, padding='VALID', scope='pool3')

    net = slim.conv2d(net, 128, 2, stride=1, padding='VALID', scope='conv4')
    net = prelu(net, scope='prelu4')

    net = slim.conv2d(net, 256, 3, stride=1, padding='VALID', scope='conv5')
    net = prelu(net, scope='prelu5')

    prob = slim.conv2d(net, 2, 1, stride=1, padding='VALID', scope='conv6-1')
    prob = slim.flatten(prob)
    prob = tf.nn.softmax(prob, axis=1, name='prob1')

    regress = slim.conv2d(net, 4, 1, stride=1, padding='VALID', scope='conv6-2')
    regress = slim.flatten(regress)

    landmark = slim.conv2d(net, 10, 1, stride=1, padding='VALID', scope='conv6-3')
    landmark = slim.flatten(landmark)
    return prob, regress, landmark


def common_arg(data_format):
    conv_scope = slim.arg_scope(
        [slim.conv2d],
        activation_fn=None)
    fc_scope = slim.arg_scope(
        [slim.fully_connected],
        activation_fn=None)
    data_format_scope = slim.arg_scope(
        [slim.conv2d, slim.max_pool2d, prelu],
        data_format=data_format)
    with conv_scope, fc_scope, data_format_scope as scope:
        return scope


def assign_from_caffe(proto, caffemodel, scope):
    caffe.set_mode_cpu()
    cnet = caffe.Net(proto, caffemodel, caffe.TEST)

    assign_ops = []
    for name, layer in cnet.layer_dict.items():
        print('layer:', name)
        param_names = ['weights', 'biases']
        n_param = len(layer.blobs)
        param_names = param_names[:n_param]

        transform = [None for i in range(n_param)]
        if layer.type == 'Convolution':
            transform = [(conv_t, None), (conv_b_t, None)]
        if layer.type == 'InnerProduct':
            transform = [(fc2conv, None), (conv_b_t, None)]
        elif layer.type == 'PReLU':
            transform = [(lambda var, v, args: v.reshape(var.get_shape().as_list()), None)]

        assert len(param_names) == len(transform)

        for i, (p, t) in enumerate(zip(param_names, transform)):
            var, = slim.get_model_variables(scope + '/' + name, p)
            print(var.name)
            v = layer.blobs[i].data.copy()
            if t:
                v = t[0](var, v, t[1])
            assign_ops.append(tf.assign(var, v))

    assert len(slim.get_model_variables(scope)) == len(assign_ops)

    return assign_ops

def regress_box(bbox, reg):
    hw = bbox[:, 2:] - bbox[:, :2]
    hw = tf.concat([hw, hw], axis=1)
    bbox = bbox + hw * reg
    return bbox

def square_box(bbox):
    hw = bbox[:, 2:] - bbox[:, :2] + 1
    max_side = tf.reduce_max(hw, axis=1, keepdims=True)
    delta = tf.concat([(hw - max_side) * 0.5, (hw - max_side) * -0.5], axis=1)
    bbox = bbox + delta
    return bbox

def stage_one(images, min_size, factor, thresold, scope):
    img_shape = tf.shape(images)
    width, height = tf.to_float(img_shape[2]), tf.to_float(img_shape[1])
    min_side = tf.to_float(tf.minimum(width, height))

    with tf.device('/cpu:0'):
        prob_arr = tf.TensorArray(
            tf.float32, size=0, clear_after_read=True,
            dynamic_size=True, element_shape=[None], infer_shape=False)
        reg_arr = tf.TensorArray(
            tf.float32, size=0, clear_after_read=True,
            dynamic_size=True, element_shape=[None, 4], infer_shape=False)
        box_arr = tf.TensorArray(
            tf.float32, size=0, clear_after_read=True,
            dynamic_size=True, element_shape=[None, 4], infer_shape=False)

    stride = 2
    cell_size = 12

    def body(i, scale, prob_arr, reg_arr, box_arr):

        width_scaled = tf.to_int32(width * scale)
        height_scaled = tf.to_int32(height * scale)
        img = tf.image.resize_bilinear(images, [height_scaled, width_scaled])
        prob, reg = det1(img, 'NHWC')

        with tf.device('/cpu:0'):
            prob, reg = prob[0], reg[0]

            scope.reuse_variables()
            mask = prob[:, :, 1] > thresold
            indexes = tf.where(mask)

            bbox = [
                tf.to_float(indexes*stride + 1)/scale,
                tf.to_float(indexes*stride + cell_size)/scale]
            bbox = tf.concat(bbox, axis=1)
            prob = tf.boolean_mask(prob[:, :, 1], mask)
            reg = tf.boolean_mask(reg, mask)

            idx = tf.image.non_max_suppression(bbox, prob, 1000, 0.5)

            bbox = tf.gather(bbox, idx)
            prob = tf.gather(prob, idx)
            reg = tf.gather(reg, idx)

            prob_arr = prob_arr.write(i, prob)
            reg_arr = reg_arr.write(i, reg)
            box_arr = box_arr.write(i, bbox)
        return i+1, scale * factor, prob_arr, reg_arr, box_arr

    _, _, prob_arr, reg_arr, box_arr = tf.while_loop(
        lambda i, scale, prob_arr, reg_arr, box_arr: min_side * scale > 12.,
        body,
        [0, 12. / min_size, prob_arr, reg_arr, box_arr],
        back_prop=False)

    prob, reg, bbox = prob_arr.concat(), reg_arr.concat(), box_arr.concat()

    idx = tf.image.non_max_suppression(bbox, prob, 1000, 0.7)
    bbox, prob, reg = tf.gather(bbox, idx), tf.gather(prob, idx), tf.gather(reg, idx)

    bbox = regress_box(bbox, reg)
    bbox = square_box(bbox)
    return bbox, prob

def stage_two(images, bbox, prob, threshold, scope):
    img_shape = tf.shape(images)
    width, height = tf.to_float(img_shape[2]), tf.to_float(img_shape[1])
    bbox_norm = bbox / [height, width, height, width]
    img_batch = tf.image.crop_and_resize(images, bbox_norm, tf.tile([0], tf.shape(prob)), [24, 24])
    prob, reg = det2(img_batch)

    with tf.device('/cpu:0'):
        mask = prob[:, 1] > threshold
        prob, reg, bbox = (
            tf.boolean_mask(prob[:, 1], mask),
            tf.boolean_mask(reg, mask),
            tf.boolean_mask(bbox, mask))
        
        idx = tf.image.non_max_suppression(bbox, prob, 1000, 0.7)
        bbox, prob, reg = tf.gather(bbox, idx), tf.gather(prob, idx), tf.gather(reg, idx)

        bbox = regress_box(bbox, reg)
        bbox = square_box(bbox)
    return bbox, prob

def stage_three(images, bbox, prob, threshold, scope):
    img_shape = tf.shape(images)
    width, height = tf.to_float(img_shape[2]), tf.to_float(img_shape[1])

    bbox_norm = bbox / [height, width, height, width]
    img_batch = tf.image.crop_and_resize(images, bbox_norm, tf.tile([0], tf.shape(prob)), [48, 48])

    prob, reg, landmarks = det3(img_batch)

    with tf.device('/cpu:0'):
        mask = prob[:, 1] > threshold
        prob, reg, bbox, landmarks = (
            tf.boolean_mask(prob[:, 1], mask),
            tf.boolean_mask(reg, mask),
            tf.boolean_mask(bbox, mask),
            tf.boolean_mask(landmarks, mask))
        
        hw = bbox[:, 2:] - bbox[:, :2]
        hw = tf.reshape(tf.tile(tf.expand_dims(hw, 2), [1, 1, 5]), [-1, 10])
        top_left = tf.reshape(tf.tile(tf.expand_dims(bbox[:, :2], 2), [1, 1, 5]), [-1, 10])
        landmarks = top_left + hw * landmarks

        bbox = regress_box(bbox, reg)
        idx = tf.image.non_max_suppression(bbox, prob, 1000, 0.6)
        bbox, prob, reg, landmarks = (
            tf.gather(bbox, idx), tf.gather(prob, idx),
            tf.gather(reg, idx), tf.gather(landmarks, idx))
    return bbox, prob, landmarks

def main(args):
    data_format = 'NHWC'
    orig_graph = tf.Graph()
    with orig_graph.as_default(), slim.arg_scope(common_arg(data_format)):

        assign_ops = []
        input_ops = []
        output_ops = []

        print('-'*80)
        print('parse det1...')

        if data_format == 'NHWC':
            img_shape = [None, None, 3]
        else:
            img_shape = [3, None, None]

        images = tf.placeholder(
            tf.float32, shape=img_shape, name='input')
        images = tf.expand_dims(images, 0)
        images = (images - 127.5) / 128.

        min_size = tf.placeholder(tf.float32, shape=[], name='min_size')
        thresholds = tf.placeholder(tf.float32, shape=[3], name='thresholds')
        factor = tf.placeholder(tf.float32, shape=[], name='factor')

        input_ops.extend([images, min_size, thresholds, factor])

        # det1
        with tf.variable_scope('pnet') as scope:
            bbox, prob = stage_one(images, min_size, factor, thresholds[0], scope)
            assign_ops.extend(
                assign_from_caffe(
                    os.path.join(args.model_dir, 'det1.prototxt'),
                    os.path.join(args.model_dir, 'det1.caffemodel'),
                    scope.name)
            )

        print('-'*80)
        print('parse det2...')
        # det2
        with tf.variable_scope('rnet') as scope:
            bbox, prob = tf.cond(
                tf.shape(bbox)[0] > 0,
                lambda : stage_two(images, bbox, prob, thresholds[1], scope),
                lambda : (tf.constant(np.zeros([0, 4], dtype='float32')),
                          tf.constant(np.zeros([0], dtype='float32')))
            )
            assign_ops.extend(
                assign_from_caffe(
                    os.path.join(args.model_dir, 'det2.prototxt'),
                    os.path.join(args.model_dir, 'det2.caffemodel'),
                    scope.name)
            )

        print('-'*80)
        print('parse det3...')
        with tf.variable_scope('onet') as scope:
            bbox, prob, landmarks = tf.cond(
                tf.shape(bbox)[0] > 0,
                lambda : stage_three(images, bbox, prob, thresholds[2], scope),
                lambda : (tf.constant(np.zeros([0, 4], dtype='float32')),
                          tf.constant(np.zeros([0], dtype='float32')),
                          tf.constant(np.zeros([0, 10], dtype='float32')))
            )
            assign_ops.extend(
                assign_from_caffe(
                    os.path.join(args.model_dir, 'det3.prototxt'),
                    os.path.join(args.model_dir, 'det3.caffemodel'),
                    scope.name)
            )

        output_ops.extend([tf.identity(prob, 'prob'), tf.identity(landmarks, 'landmarks'), tf.identity(bbox, 'box')])
        init_op = tf.global_variables_initializer()

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(graph=orig_graph, config=config)

    sess.run(init_op)
    sess.run(assign_ops)

    orig_graph_def = orig_graph.as_graph_def()
    dst_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, orig_graph_def, [t.op.name for t in output_ops])

    with open(args.dst, 'wb') as f:
        f.write(dst_graph_def.SerializeToString())

if __name__ == '__main__':
    parser = argparse.ArgumentParser('parse mtcnn caffe model to tensorflow')
    parser.add_argument('--model_dir',default="../model/caffe", help='directory contain mtcnn caffe model.')
    parser.add_argument('--dst', default="mtcnn.pb",help='the dst tensorflow model')
    args = parser.parse_args()
    main(args)
