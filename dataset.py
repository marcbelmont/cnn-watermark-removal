from PIL import Image
from glob import glob
import pickle
import tensorflow as tf
import numpy as np

IMG_SIZE = 120
FLAGS = None
DEBUG = None


def batch_masks(global_step, height, width, min_opacity, max_opacity):
    return tf.concat([
        tf.expand_dims(create_mask(
            global_step, height, width, min_opacity, max_opacity), 0)
        for _ in range(FLAGS.batch_size)], 0)


def create_mask(global_step, height, width, min_opacity, max_opacity):
    if global_step:
        min_opacity = tf.train.polynomial_decay(
            max_opacity, global_step,
            decay_steps=60000, end_learning_rate=min_opacity)

    mask_h = tf.random_uniform([], int(height * .7), int(height * .9), tf.int32)
    mask_w = tf.random_uniform([], int(width * .1), int(width * .3), tf.int32)
    opacity = tf.random_uniform([], min_opacity, max_opacity, tf.float32)
    max_angle = tf.random_uniform([], -1.5, 1.5, tf.float32)

    mask = tf.ones([mask_h, mask_w]) * opacity
    mask *= tf.cast(tf.random_uniform([], 0, 2, tf.int32) * 2 - 1, tf.float32)
    y_pos = tf.random_uniform([], 0, height - mask_h, tf.int32)
    x_pos = tf.random_uniform([], 0, width - mask_w, tf.int32)
    mask = tf.pad(mask, [[y_pos, height - mask_h - y_pos],
                         [x_pos, width - mask_w - x_pos]])  # Costly
    mask = tf.expand_dims(mask, 2)
    mask.set_shape([height, width, 1])
    mask = tf.contrib.image.rotate(
        mask, tf.random_uniform([], -max_angle, max_angle, tf.float32))  # Costly
    return mask


def dataset_paths(paths):
    def load_image(path):
        image = tf.image.decode_image(tf.read_file(path))
        image = tf.cast(image, tf.float32) / 255
        image = tf.image.resize_image_with_crop_or_pad(image, 300, 300)
        image.set_shape([300, 300, 3])
        return image
    dataset = tf.Dataset.from_tensor_slices(tf.constant(paths))
    dataset = dataset.map(load_image)
    dataset = dataset.batch(FLAGS.batch_size)
    dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.repeat()
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
    return next_element, lambda _: None


def get_records():
    records = glob('data/*.tfrecords')
    if not records:
        convert_to_record()
        records = glob('data/*.tfrecords')
    return records


def dataset_split(dataset_fn, split):
    records = get_records()
    split = int(len(records) * split)
    train, val = dataset_fn(records[:split]), dataset_fn(records[split:])
    iterator = tf.data.Iterator.from_structure(
        train.output_types, train.output_shapes)
    next_element = iterator.get_next()
    return (next_element,
            [iterator.make_initializer(x) for x in [train, val]])


def dataset_voc2012():
    records = get_records()
    dataset = dataset_voc2012_rec(records)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
    return next_element, lambda x: None


def dataset_voc2012_rec(records):

    def parse_function(serialized):
        features = tf.parse_single_example(serialized, features=dict(
            height=tf.FixedLenFeature([], tf.int64),
            width=tf.FixedLenFeature([], tf.int64),
            image_raw=tf.FixedLenFeature([], tf.string)))
        image = tf.reshape(tf.decode_raw(features['image_raw'], tf.uint8),
                           [tf.cast(features['height'], tf.int32),
                            tf.cast(features['width'], tf.int32),
                            3])
        image = tf.cast(image, tf.float32) / 255
        image = tf.cond(
            tf.logical_and(tf.greater(tf.shape(image)[0], IMG_SIZE),
                           tf.greater(tf.shape(image)[1], IMG_SIZE)),
            lambda: tf.random_crop(image, [IMG_SIZE, IMG_SIZE, 3]),
            lambda: tf.image.resize_image_with_crop_or_pad(image, IMG_SIZE, IMG_SIZE),)
        image.set_shape([IMG_SIZE, IMG_SIZE, 3])
        image = tf.image.random_flip_left_right(image)
        return image

    dataset = tf.data.TFRecordDataset(records)
    dataset = dataset.map(parse_function)
    dataset = dataset.batch(FLAGS.batch_size)
    dataset = dataset.filter(
        lambda batch: tf.equal(tf.shape(batch)[0], FLAGS.batch_size))
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.repeat()
    return dataset


def dataset_cifar():
    images = standardize(get_images())
    images_placeholder = tf.placeholder(tf.float32, images.shape)
    dataset = tf.data.Dataset.from_tensor_slices(images_placeholder)
    dataset = dataset.map(tf.image.random_flip_left_right)
    dataset = dataset.batch(FLAGS.batch_size)
    dataset = dataset.filter(
        lambda batch: tf.equal(tf.shape(batch)[0], FLAGS.batch_size))
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.repeat()
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    def init(sess):
        sess.run(iterator.initializer, feed_dict={images_placeholder: images})
    return next_element, init


def get_images():
    data = None
    for path in glob('cifar-10-batches-py/data_batch_*'):
        with open(path, 'rb') as fo:
            binary = pickle.load(fo, encoding='bytes')
        if data is None:
            data = binary[b'data']
        else:
            data = np.concatenate((data, binary[b'data']))
        if DEBUG:
            break
    data = data.reshape((-1, 3, 32, 32)).transpose((0, 2, 3, 1))
    return data


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_record():
    print('Creating TFRecords')
    filenames = glob('data/VOCdevkit/VOC2012/JPEGImages/*.jpg')
    if not filenames:
        print('Source images are missing!')
    writer = None
    for i, filename in enumerate(filenames):
        if i % 200 == 0:
            print(i)
            if writer:
                writer.close()
            # Recommanded size is 100mb
            writer = tf.python_io.TFRecordWriter('data/voc-%s.tfrecords' % i)
        image = np.array(Image.open(filename))
        example = tf.train.Example(features=tf.train.Features(
            feature=dict(
                height=_int64_feature(image.shape[0]),
                width=_int64_feature(image.shape[1]),
                image_raw=_bytes_feature(image.tostring()))))
        writer.write(example.SerializeToString())
    writer.close()


def standardize(batch):
    return (batch.astype(np.float32) / 255)


def unstandardize(batch):
    return (batch * 255).astype(np.uint8)
