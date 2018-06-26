# -*- coding: utf-8 *-*

# See https://www.tensorflow.org/get_started/eager

import os
import time
from enum import Enum

start = time.time()
import tensorflow as tf
import tensorflow.contrib.eager as tfe

# Enable eager execution (because I'm sooo impatient!)
tf.enable_eager_execution()
stop = time.time()

print("TensorFlow version: " + tf.VERSION)
print("Eager execution: {0} (took {1:.2f}s)".format(str(tf.executing_eagerly()), stop - start))

### Iris Classification Problem ###

# Download the dataset

train_dataset_url = "http://download.tensorflow.org/data/iris_training.csv"

train_dataset_local_file_path = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                                        origin=train_dataset_url)

print("A copy of the iris training dataset has been downloaded at from {0} to {1}.".format(train_dataset_url, train_dataset_local_file_path))


class Iris(Enum):
	"""
	Enumerate the type of Iris for the training dataset. Each enum has an integer value which match the tutorial
	"""
	SETOSA = 0
	VERSICOLOR = 1
	VIRGINICA = 2


def parse_csv_line(line: str) -> tuple:
	"""
	Parse one line of a CSV file using tensorflow.
	:param line: The line of the CSV. It must contains five floats, separated by commas.
	:return: Return the features and the label of the line
	"""
	template = [[0.], [0.], [0.], [0.], [0.]]
	parsed_line = tf.decode_csv(line, template)
	
	features = tf.reshape(parsed_line[:-1], shape=(4,))
	label = tf.reshape(parsed_line[-1], shape=())
	
	return features, label


# Load the CSV file (line by line)
training_dataset: tf.data.Dataset = tf.data.TextLineDataset(train_dataset_local_file_path)

# Skip the header
training_dataset = training_dataset.skip(1)

# Parse each row using the function 'parse_csv_line'
training_dataset = training_dataset.map(parse_csv_line)

# Shuffle the entries (work best)
training_dataset = training_dataset.shuffle(buffer_size=1000)

# The number of examples in the batch is 32
training_dataset = training_dataset.batch(batch_size=32)

# Print an example from a batch
features, label = iter(training_dataset).__next__()
print("Example from a batch:\n\tfeatures = {}\n\tlabel = {}".format(str(features[0]), str(label[0])))

# Stop my training at https://www.tensorflow.org/get_started/eager#select_the_type_of_model
