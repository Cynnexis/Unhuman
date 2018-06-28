# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from unhuman.utils.Stopwatch import Stopwatch
import argparse
import numpy as np

import tensorflow as tf

IMAGE_WIDTH_PX = 28
IMAGE_HEIGHT_PX = 28
IMAGE_SIZE_PX = IMAGE_WIDTH_PX * IMAGE_HEIGHT_PX


def cnn_model(features: dict, labels, mode: tf.estimator.ModeKeys = tf.estimator.ModeKeys.TRAIN) \
	-> tf.estimator.EstimatorSpec:
	"""
	Model function for CNN.
	:param features: The given features (supervised training)
	:param labels: The given labels (supervised training)
	:param mode: A ModeKeys
	:return: An estimator
	"""
	
	# Create input layer
	input_layer = tf.reshape(tensor=features['x'], shape=[-1, 28, 28, 1])
	
	# Create Convolutional Layer n°1
	conv1 = tf.layers.conv2d(
		inputs=input_layer,
		filters=32,
		kernel_size=[5, 5],
		padding="same",
		activation=tf.nn.relu
	)
	
	# Create Pooling Layer n°1
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
	
	# Create Convolutional Layer n°2
	conv2 = tf.layers.conv2d(
		inputs=pool1,
		filters=64,
		kernel_size=[5, 5],
		padding="same",
		activation=tf.nn.relu
	)
	
	# Create Pooling Layer n°2
	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
	
	# Create the Dense Layer
	pool2_flat = tf.reshape(tensor=pool2, shape=[-1, 7 * 7 * 64])
	dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
	dropout = tf.layers.dropout(
		inputs=dense,
		rate=0.4,
		training=mode == tf.estimator.ModeKeys.TRAIN
	)
	
	# Create Logits Layer
	logits= tf.layers.dense(inputs=dropout, units=10)
	
	# Create the prediction structure
	predictions = {
		# Generate predictions (for PREDICT and EVAL mode)
		"classes": tf.argmax(input=logits, axis=1),
		# Add `softmax_tensor` to the graph. It is used for PREDICT and by the `logging_hook`.
		"probabilities": tf.nn.softmax(logits=logits, name="softmax_tensor")
	}
	
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
	
	# Calculate error (for TRAIN or EVAL modes)
	error = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
	
	# Configure the training optimizer
	
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		train_op = optimizer.minimize(loss=error, global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=error, train_op=train_op)
	
	# Add evaluation metrics for EVAL mode
	eval_metrics_ops = {
		"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
	}
	
	return tf.estimator.EstimatorSpec(mode=mode, loss=error, eval_metric_ops=eval_metrics_ops)


def main(argv):
	mnist = tf.contrib.learn.datasets.load_dataset("mnist")
	pass


if __name__ == "__main__":
	tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run(main)
