import tensorflow as tf
import numpy as np
import os
import watermarks as w
from glob import glob
from datetime import datetime

w.DEBUG = True


class WatermarkTest(tf.test.TestCase):

    def setUp(self):
        w.FLAGS.logdir = '/tmp/tensorflow_log/%s' % datetime.now()
        w.FLAGS.logdir = '/tmp/tensorflow_log'
        w.FLAGS.batch_size = 4
        w.FLAGS.learning_rate = 1e-1
        [os.remove(x) for x in glob(w.FLAGS.logdir + '/events*')]

    ########
    # Data #
    ########

    def test_dataset_paths(self):
        w.FLAGS.batch_size = 2
        with self.test_session() as sess:
            np.set_printoptions(threshold=np.nan)
            x, _ = w.dataset_paths(['data/cat.png'])
            sess.run(tf.tables_initializer())
            self.assertTupleEqual(x.eval().shape, (1, 300, 300, 4))

    def test_dataset_mask(self):
        w.FLAGS.batch_size = 2
        with self.test_session():
            np.set_printoptions(threshold=np.nan)
            mask = w.batch_masks(None, 32, 32, .1, .4)
            self.assertTupleEqual(mask.eval().shape, (2, 32, 32, 1))

    def test_dataset_cifar(self):
        w.FLAGS.batch_size = 2
        with self.test_session() as sess:
            np.set_printoptions(threshold=np.nan)
            x, init = w.dataset_cifar()
            init(sess)
            self.assertTupleEqual(x.eval().shape, (2, 32, 32, 3))

    def test_dataset_voc2012(self):
        w.FLAGS.batch_size = 2
        with self.test_session() as sess:
            x, init = w.dataset_voc2012()
            sess.run(tf.tables_initializer())
            self.assertTupleEqual(x.eval().shape, (2, 120, 120, 3))

    ############
    # Pipeline #
    ############

    def test_training(self):
        with self.test_session() as sess:
            w.train(sess, w.dataset_voc2012)

    def test_inference_voc(self):
        with self.test_session() as sess:
            dv = w.dataset_voc2012
            results = w.inference(sess, dv, 1)
            self.assertTupleEqual(results[0].shape, (4, 120, 120, 3))

    def test_inference_other(self):
        with self.test_session() as sess:
            d_cherry = lambda: w.dataset_paths(['data/cat.png', ])
            dm = lambda: w.dataset_paths(['data/cat-mask.png', ])
            ds = lambda: w.dataset_paths(['data/cat-selection.png', ])
            results = w.inference(sess, d_cherry, 1, dm, ds)
            self.assertTupleEqual(results[0].shape, (1, 300, 300, 3))


if __name__ == '__main__':
    tf.test.main()
