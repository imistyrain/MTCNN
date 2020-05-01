import argparse

import tensorflow as tf

def preprocess(path):
    # read file from disk
    raw_str = tf.read_file(path)
    
    # decode image
    img = tf.image.decode_image(raw_str, 3)
    img.set_shape([None, None, 3])
    
    # convert image from RGB to BGR
    img = tf.reverse(img, [2])
    img = tf.to_float(img)

    # import mtcnn model
    with open('./mtcnn.pb', 'rb') as f:
        gd = tf.GraphDef.FromString(f.read())
        prob, landmarks, box = tf.import_graph_def(
            gd,
            {'input:0': img, 'min_size:0': tf.constant(40.),
             'factor:0': tf.constant(0.709),
             'thresholds:0': tf.constant([0.6,0.7,0.7])},
            ['prob:0', 'landmarks:0', 'box:0'],
            name='')
    return path, prob, landmarks, box


def main(args):

    with tf.device('/cpu:0'):
        # build dataset
        dataset = (tf.data
            .TextLineDataset(args.imglist)
            .map(preprocess, 8)             # processing data with multi-threads
            .prefetch(1))                   # prefetch in other thread
        iterator = dataset.make_one_shot_iterator()
        path, prob, landmarks, box = iterator.get_next()
    
    # build session
    config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False,
        intra_op_parallelism_threads=16, inter_op_parallelism_threads=16)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    # handle detection results
    dst_list = open(args.dst + '.list', 'w')
    dst_pts = open(args.dst + '.pts', 'w')
    try:
        while True:
            p, lm = sess.run([path, landmarks])

            if len(lm) == 0:
                continue

            dst_list.write('{}\n'.format(p.decode()))
            for i, l in enumerate(lm[0]):
                dst_pts.write('{:.4f}'.format(l))
                dst_pts.write(',' if i < 9 else '\n')
    except tf.errors.OutOfRangeError as e:
        pass

    dst_list.close()
    dst_pts.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tensorflow mtcnn')
    parser.add_argument('imglist', help='image list')
    parser.add_argument('dst', help='dst prefix')
    args = parser.parse_args()
    main(args)
