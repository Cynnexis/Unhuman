# -*- coding: utf-8 *-*
# Code duplication from eager.py, but minimize

import os
import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

train_dataset_url = "http://download.tensorflow.org/data/iris_training.csv"
test_dataset_url = "http://download.tensorflow.org/data/iris_test.csv"

train_dataset_local_file_path = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                                        origin=train_dataset_url)
test_dataset_local_file_path  = tf.keras.utils.get_file(fname=os.path.basename(test_dataset_url),
                                                        origin=test_dataset_url)


def parse_csv_line(line: str) -> tuple:
	template = [[0.], [0.], [0.], [0.], [0]]
	parsed_line = tf.decode_csv(line, template)
	features = tf.reshape(parsed_line[:-1], shape=(4,))
	label = tf.reshape(parsed_line[-1], shape=())
	return features, label


training_dataset = tf.data.TextLineDataset(train_dataset_local_file_path)
training_dataset = training_dataset.skip(1)
training_dataset = training_dataset.map(parse_csv_line)
training_dataset = training_dataset.shuffle(buffer_size=1000)
training_dataset = training_dataset.batch(batch_size=32)

model = tf.keras.Sequential([
	tf.keras.layers.Dense(units=10, activation="relu", input_shape=(4,)),
	tf.keras.layers.Dense(units=10, activation="relu"),
	tf.keras.layers.Dense(units=3)
])


def grad(model: tf.keras.models.Model, inputs, targets):
	with tf.GradientTape() as tape:
		error_value = tf.losses.sparse_softmax_cross_entropy(labels=targets, logits=model(inputs))
	return tape.gradient(error_value, model.variables)


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

for iteration in range(200+1):
	for x, y in training_dataset:
		grads = grad(model, x, y)
		optimizer.apply_gradients(zip(grads, model.variables), global_step=tf.train.get_or_create_global_step())

test_dataset = tf.data.TextLineDataset(test_dataset_local_file_path)
test_dataset = test_dataset.skip(1)
test_dataset = test_dataset.map(parse_csv_line)
test_dataset = test_dataset.shuffle(1000)
test_dataset = test_dataset.batch(32)

# Evaluation

test_accuracy = tfe.metrics.Accuracy()

for x, y in test_dataset:
	prediction = tf.argmax(model(x), axis=1, output_type=tf.int32)
	test_accuracy(prediction, y)

print("Test set accuracy: {:.3%}".format(test_accuracy.result()))
