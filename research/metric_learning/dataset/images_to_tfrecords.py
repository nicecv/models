#coding: utf-8
from collections import defaultdict
import os
import tensorflow as tf
import random

tf.flags.DEFINE_string(
    'train_image_dir', '/raid/data/nice/metric_data/crop_images/train_crop_images/shoes', 'train images directory')
tf.app.flags.DEFINE_string(
    'val_image_dir', '/raid/data/nice/metric_data/crop_images/val_crop_images/shoes', 'validation images directory')
tf.app.flags.DEFINE_string(
    'train_tfrecord_dir', '/raid/data/nice/metric_data/tfrecords/train/shoe', 'Path to ')
tf.app.flags.DEFINE_string(
    'val_tfrecord_dir', '/raid/data/nice/metric_data/tfrecords/val/shoe', 'Path to')
tf.app.flags.DEFINE_string(
    'label_dir', '/raid/data/nice/metric_data/crop_images/query/huarache', 'path to label file')
tf.app.flags.DEFINE_integer('train_num_batches', 500,
                            'Number of batches.')
tf.app.flags.DEFINE_integer('val_num_batches', 10,
                            'Number of batches.')

FLAGS = tf.app.flags.FLAGS

def main(image_dir, label_dir, tfrecord_dir, numbatches, is_training=True):
    labelname2images = defaultdict(list)
    for dirname in os.listdir(image_dir):
        for filename in os.listdir(os.path.join(image_dir, dirname)):
            image_path = os.path.join(image_dir, dirname, filename).decode('utf-8')
            labelname2images[dirname].append(image_path)
    
    if is_training:
        labels = range(len(labelname2images.keys()))
        label2labelname = dict(zip(labels,labelname2images.keys()))
        with open(os.path.join(label_dir, 'label.txt'), 'w') as fw:
            for label,labelname in label2labelname.items():
                fw.write(labelname + '\n')
    else:
        labelnames = [line.strip() for line in open(os.path.join(label_dir, 'label.txt'), 'r')]
        labels = range(len(labelnames))
        label2labelname = dict(zip(labels, labelnames))

    batch_examples = []
    batch_size = 512
    batch_index = 0
    
    while batch_index < numbatches:
        label = random.choice(labels)
        labelname = label2labelname[label]
        image_paths = random.sample(labelname2images[labelname], 2)    
        image0 = open(image_paths[0], 'r').read()
        image1 = open(image_paths[1], 'r').read()
        feature_dict = {
            'image0': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image0])),
            'image1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image1])),
            'format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['jpeg'])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        batch_examples.append(example.SerializeToString())
        if len(batch_examples) == batch_size:
            batch_index += 1
            writer = tf.python_io.TFRecordWriter(os.path.join(tfrecord_dir, '%.5d-of-%.5d' % (batch_index,numbatches)))
            for example_string in batch_examples:
                writer.write(example_string)
            writer.close()
            batch_examples = []


if __name__ == '__main__':
    main(FLAGS.train_image_dir, FLAGS.label_dir, FLAGS.train_tfrecord_dir, FLAGS.train_num_batches, True)
    main(FLAGS.val_image_dir, FLAGS.label_dir, FLAGS.val_tfrecord_dir, FLAGS.val_num_batches, False)