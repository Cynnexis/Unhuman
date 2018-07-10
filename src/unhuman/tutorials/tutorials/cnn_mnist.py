# -*- coding: utf-8 -*-

"""
This tutorial comes from https://www.tensorflow.org/tutorials/layers
and https://www.tensorflow.org/versions/r1.2/get_started/mnist/beginners
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from unhuman.tutorials.getstarted import logdir
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
	"""
	The shape is [batch_size, image width, image height, channels], where :
	* batch_size is the size of subset of examples during the training (here -1  means 'computed automatically'
	* channels is the number of channels in the image. Here it is 1 (black).
	"""
	input_layer = tf.reshape(tensor=features['x'], shape=[-1, IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX, 1])
	
	# Create Convolutional Layer n°1
	"""
	* The filters means the number of filters to apply on the sub-image
	* The kernel size is the dimension of the sub-image (here, 5px * 5px)
	* "padding" is either 'valid' or 'same' (the output tensor should have the same height and width values as the input
	tensor)
	"""
	conv1 = tf.layers.conv2d(
		inputs=input_layer,
		filters=32,
		kernel_size=[5, 5],
		padding="same",
		activation=tf.nn.relu
	)
	
	# Create Pooling Layer n°1
	"""
	* pool_size: The size of the max pooling filter, here 2px * 2px
	* strides: The subregions extracted by the filter should be separated by 2px.
	The return value is a layer with a shape of [batch_size, img width / 2, img height / 2, channels * 32]
	=> [-1, 14, 14, 32]
	"""
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
	"""
	Return shape [batch_size, 7, 7, 64]
	"""
	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
	
	# Create the Dense Layer
	"""
	Each example has 7 (pool2 height) * 7 (pool2 width) * 64 (pool2 channels) features = 3'136 features
	"""
	pool2_flat = tf.reshape(tensor=pool2, shape=[-1, 7 * 7 * 64])
	
	"""
	* units: the number of neurons in this layer (here 1024)
	"""
	dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
	
	"""
	The dropout layer helps to regularize the dense layer to improve the results
	* rate: 40% of the elements will be randomly dropped out during training
	"""
	dropout = tf.layers.dropout(
		inputs=dense,
		rate=0.4,
		training=mode == tf.estimator.ModeKeys.TRAIN
	)
	
	# Create Logits Layer
	"""
	The final layer which will give the results (each output is associated to a probability)
	* units: The number of neurons, outputs
	"""
	logits= tf.layers.dense(inputs=dropout, units=10)
	
	# Create the prediction structure
	"""
	'predictions' is a structure which will take the output given by 'logits' and will analyse precisely the values
	and tell which one has the highest probability, and the probabilities attached to each digits.
	"""
	predictions = {
		# Generate predictions (for PREDICT and EVAL mode)
		# Gives the index of the element with the highest probability (from 0 to 9)
		"classes": tf.argmax(input=logits, axis=1),
		
		
		# Add `softmax_tensor` to the graph. It is used for PREDICT and by the `logging_hook`.
		# Fetch the probabilities attached to each digits
		# * name: Reference that will be used later for logging
		"probabilities": tf.nn.softmax(logits=logits, name="softmax_tensor")
	}
	
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
	
	# Calculate error (for TRAIN or EVAL modes)
	"""
	For more information about cross entropy, see https://en.wikipedia.org/wiki/Cross_entropy
	"""
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


# noinspection PyUnusedLocal
def main(unused_argv):
	mnist = tf.contrib.learn.datasets.load_dataset("mnist")
	training_data = mnist.train.images
	
	"""
	Convert 'mnist.train.labels' to a numpy array
	See https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html
	"""
	training_labels = np.asarray(a=mnist.train.labels, dtype=np.int32)
	eval_data = mnist.test.images
	eval_labels = np.asarray(a=mnist.test.labels, dtype=np.int32)
	
	# Create the Estimator
	mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model, model_dir="/tmp/mnist_convolutional_model")
	
	# Setup logging for prediction
	tensors_to_log = {
		"probabilities": "softmax_tensor"
	}
	logging_hook = tf.train.LoggingTensorHook(
		tensors=tensors_to_log,
		every_n_iter=50
	)
	
	# Train the model
	training_inputs = tf.estimator.inputs.numpy_input_fn(
		x={'x': training_data},
		y=training_labels,
		batch_size=100,
		num_epochs=None,
		shuffle=True
	)
	
	# The number of step is normally 20000, but the computer would take about 1h to compute it, so we decrease it to 1000
	mnist_classifier.train(
		input_fn=training_inputs,
		steps=10,
		hooks=[logging_hook]
	)
	
	# Evaluate the model
	eval_input = tf.estimator.inputs.numpy_input_fn(
		x={'x': eval_data},
		y=eval_labels,
		num_epochs=1,
		shuffle=False
	)
	eval_results = mnist_classifier.evaluate(input_fn=eval_input)
	tf.summary.text(name="eval_results", tensor=tf.constant(str(eval_results)))
	
	# Print results
	print(eval_results)


if __name__ == "__main__":
	g = tf.Graph()
	with g.as_default():
		with tf.Session() as sess:
			write = tf.summary.FileWriter(logdir=logdir, graph=sess.graph)
			
			tf.logging.set_verbosity(tf.logging.INFO)
			tf.app.run(main)
else:
	print("Found __name__: " + __name__)
