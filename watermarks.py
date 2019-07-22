from dataset import batch_masks, unstandardize, dataset_paths, dataset_voc2012, dataset_split, dataset_voc2012_rec  # noqa
from io import BytesIO
from time import time
import IPython.display
import PIL.Image
import dataset
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

tf.flags.DEFINE_string('logdir', None, 'Log directory')
tf.flags.DEFINE_integer("batch_size", 32, "Batch size")
tf.flags.DEFINE_float("learning_rate", .005, "Learning rate")
tf.flags.DEFINE_string("dataset", 'dataset_voc2012_rec', "Dataset to use")
tf.flags.DEFINE_string('image', None, 'Image with watermark')
tf.flags.DEFINE_string('selection', None, 'Where to do the removal')

FLAGS = tf.flags.FLAGS
dataset.FLAGS = FLAGS
DEBUG = False
dataset.DEBUG = DEBUG

#########
# Model #
#########


def dense_block(net, growth_rate, channels_init, layers, training):
    # Dense block https://github.com/liuzhuang13/DenseNet/blob/master/models/DenseConnectLayer.lua
    # Receptive field: l * (k - 1) + k
    for i, channels in enumerate(
            [channels_init + i * growth_rate for i in range(layers)]):
        # 3x3 convolution
        previous_input = net
        net = tf.layers.batch_normalization(net, training=training)
        net = tf.nn.relu(net)

        # Bottleneck
        if channels > 4 * growth_rate:
            net = tf.layers.conv2d(
                net,
                4 * growth_rate, 1,
                padding='same',
                activation=None,)
            net = tf.layers.batch_normalization(net, training=training)
            net = tf.nn.relu(net)

        net = tf.layers.conv2d(
            net,
            growth_rate, 3,
            padding='same',
            activation=None,)
        net = tf.concat([previous_input, net], axis=3)
    return net


def selection_margin(masks, margin):
    selection = tf.nn.conv2d(masks, tf.ones([
        margin * 2 + 1, margin * 2 + 1, 1, 1]), [1, 1, 1, 1], 'SAME')
    selection = tf.clip_by_value(tf.abs(tf.ceil(selection)), 0, 1)
    return selection


def atrous_conv2d(inputs, filters, kernel_size, rate):
    shape = [kernel_size, kernel_size, inputs.shape.as_list()[-1], filters]
    initializer = tf.contrib.layers.xavier_initializer()
    weight = tf.Variable(initializer(shape))
    conv = tf.nn.atrous_conv2d(inputs, weight, rate, 'SAME')
    return conv


def model(images, training):
    growth_rate = 16
    channels_init = growth_rate * 2
    bottleneck_channels = 32

    net = tf.layers.conv2d(images, channels_init, 3,
                           padding='same', activation=None,)

    net = dense_block(net, growth_rate, channels_init, 4, training)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)

    # Bottleneck
    net = tf.layers.conv2d(
        net, bottleneck_channels, 1, padding='same', activation=None,)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)

    # Dilation layers to increase the receptive field
    # http://vladlen.info/papers/DRN.pdf
    net = atrous_conv2d(net, bottleneck_channels, 3, 2)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)

    net = atrous_conv2d(net, bottleneck_channels, 3, 4)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)

    net = atrous_conv2d(net, bottleneck_channels, 3, 2)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)

    net = tf.layers.conv2d(net, 3, 3, padding='same', activation=None)
    return net

#############
# Inference #
#############


def inference(sess, dataset, passes=1,
              dataset_mask=False, dataset_selection=False,
              min_opacity=.15, max_opacity=.4):
    # Data sources
    next_image, iterator_init = dataset()
    image_shape = next_image.shape.as_list()
    if dataset_mask:
        next_mask, iterator_init = dataset_mask()
        next_mask = next_mask[:, :, :, 0:1]
    else:
        next_mask = batch_masks(
            None, image_shape[1], image_shape[2], min_opacity, max_opacity)
    if dataset_selection:
        next_selection, iterator_init = dataset_selection()

    # Model
    images_p = tf.placeholder(tf.float32, shape=[None] + image_shape[1:])
    mask_p = tf.placeholder(tf.float32, shape=[None] + image_shape[1:3] + [1])
    selection_p = tf.placeholder(tf.float32, shape=[None] + image_shape[1:3] + [1])

    image_w = tf.clip_by_value(images_p - mask_p, 0, 1)
    selection_conv = selection_margin(mask_p, 4)
    predictions = model(image_w, False) * selection_p
    gen_mask = tf.clip_by_value(tf.abs(predictions), 0, 1)
    reconstruction = tf.clip_by_value(image_w + predictions, 0, 1)
    accuracy = get_accuracy(reconstruction, images_p)

    # Inference
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    iterator_init(sess)

    try:
        saver = tf.train.Saver()
        saver.restore(sess, "/tmp/model.ckpt")
    except Exception as e:
        print('An error occurred when trying to restore : %s' % e)

    # Pass 1
    batch = sess.run([next_image, next_mask])
    if dataset_selection:
        selection = sess.run(tf.expand_dims(next_selection[:, :, :, 0], 3))
    else:
        selection = sess.run(selection_conv, feed_dict={mask_p: batch[1]})

    feed_dict = {images_p: batch[0][:, :, :, 0:3],
                 mask_p: batch[1][:batch[0].shape[0]],
                 selection_p: selection[:batch[0].shape[0]]}
    images = sess.run([image_w, reconstruction, gen_mask, accuracy], feed_dict=feed_dict)
    results = [images[0], images[1]]
    print('Mean accuracy %.3f%%' % (images[3] * 100))

    # Pass 2
    for _ in range(1, passes):
        reconstruction1 = images[1]
        feed_dict = {images_p: reconstruction1,
                     mask_p: np.zeros(list(reconstruction1.shape[:-1]) + [1])}
        images = sess.run([image_w, reconstruction, gen_mask], feed_dict=feed_dict)
        results += [images[1]]

    # Reformat
    results += [images[2]]
    images = [unstandardize(x) for x in results]
    return images

############
# Training #
############


def train(sess, dataset, min_opacity=.15, max_opacity=.4):
    global_step = tf.Variable(0, name='global_step', trainable=False)
    training = tf.placeholder(tf.bool, shape=[])
    with tf.device('/cpu:0'):
        next_image, iterator_inits = dataset_split(dataset, .8)

    masks = batch_masks(
        global_step, next_image.shape.as_list()[1], next_image.shape.as_list()[2],
        min_opacity, max_opacity)
    image_w = tf.clip_by_value(next_image - masks, 0, 1)
    predictions = model(image_w, training) * selection_margin(masks, 4)
    tf.summary.image('masks', predictions)

    # Define loss
    image_mask = -(image_w - next_image)  # Mask after application on the image
    abs_loss = tf.losses.absolute_difference(
        predictions, image_mask, loss_collection=None)**.5
    tf.losses.add_loss(abs_loss)
    loss = tf.losses.get_total_loss(True)
    tf.summary.scalar('loss', loss)

    # Optimizer
    learning_rate = tf.train.polynomial_decay(
        FLAGS.learning_rate, global_step,
        decay_steps=60000, end_learning_rate=.0005)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(
            loss,
            global_step=global_step)

    # Training loop
    sess.run(tf.global_variables_initializer())
    sess.run(iterator_inits[0])

    saver = tf.train.Saver()
    summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(FLAGS.logdir, 'train'), sess.graph)
    val_writer = tf.summary.FileWriter(os.path.join(FLAGS.logdir, 'val'), sess.graph)

    for i in range(1, 2 if DEBUG else int(1e6)):
        if DEBUG:
            start_time = time()
            _, loss_, predictions_ = sess.run([train_op, loss, predictions],
                                              feed_dict={training: True})
            batch_time = 1000 * (time() - start_time) / FLAGS.batch_size
            print('Time %dms, Loss %f' % (batch_time, loss_))
            continue

        _, summaries_, global_step_ = sess.run(
            [train_op, summaries, global_step], feed_dict={training: True})
        train_writer.add_summary(summaries_, global_step_)

        # Save model
        if i % 2000 == 0:
            path = saver.save(sess, "/tmp/model.ckpt")
            print(i, 'Saving at', path)
            sess.run(iterator_inits[1])  # switch to validation dataset
            while True:
                try:
                    _, summaries_ = sess.run([loss, summaries],
                                             feed_dict={training: False})
                    val_writer.add_summary(summaries_, global_step_)
                except tf.errors.OutOfRangeError:
                    break
            sess.run(iterator_inits[0])
    return


###########
# Helpers #
###########

def get_accuracy(prediction, target):
    diff = tf.reduce_mean(tf.abs(prediction - target) / 255, [1, 2, 3])
    return (1 - tf.reduce_mean(diff))


def show_array(a, fmt='png'):
    a = np.uint8(a)
    f = BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    IPython.display.display(IPython.display.Image(data=f.getvalue()))


def show_images(images):
    for i in range(images.shape[0]):
        if images.shape[1] <= 32:
            plt.figure()
            plt.imshow(images[i], interpolation='nearest')
        else:
            show_array(images[i])

#######
# Run #
#######


def main(_):
    with tf.Session() as sess:
        if FLAGS.image and FLAGS.selection:
            images = inference(
                sess,
                lambda: dataset_paths([FLAGS.image]),
                0,
                lambda: dataset_paths(['assets/empty.png']),
                lambda: dataset_paths([FLAGS.selection]))
            image = np.squeeze(images[1])
            PIL.Image.fromarray(image).save('output.png')
        else:
            train(sess, globals()[FLAGS.dataset])
    return


if __name__ == '__main__':
    tf.app.run()
