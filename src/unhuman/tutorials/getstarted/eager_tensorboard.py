# -*- coding: utf-8 -*-

# See https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard

import tensorflow as tf
tf.enable_eager_execution()

from unhuman.tutorials.getstarted import logdir
from unhuman.tutorials.getstarted.eager_class import EagerClass

g = tf.Graph()
with g.as_default():
	with tf.Session() as sess:
		writer = tf.contrib.summary.create_file_writer(logdir=logdir)
		em = EagerClass.main()
