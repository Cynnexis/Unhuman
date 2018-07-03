# -*- coding: utf-8 *-*

# See https://www.tensorflow.org/get_started/eager

import os

import matplotlib.pyplot as plt

from unhuman.tutorials.getstarted.Iris import Iris
from unhuman.utils.Stopwatch import Stopwatch

sw_eager = Stopwatch(start_now=True)
import tensorflow as tf
import tensorflow.contrib.eager as tfe

# Enable eager execution (because I'm sooo impatient!)
tf.enable_eager_execution()
sw_eager.stop()

print("TensorFlow version: " + tf.VERSION)
print("Eager execution: {0} (took {1:.2f}s)".format(str(tf.executing_eagerly()), sw_eager.elapsed()))

"""
PRINT:
TensorFlow version: 1.8.0
Eager execution: True (took 14.01s)
"""

del sw_eager

### Iris Classification Problem ###

# Download the dataset: The first 4 numbers are the sepal length, sepal width, petal length and petal width. The last
# one is the label (see enum Iris)

train_dataset_url = "http://download.tensorflow.org/data/iris_training.csv"

train_dataset_local_file_path = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                                        origin=train_dataset_url)

print("A copy of the iris training dataset has been downloaded at from {0} to {1}.".format(train_dataset_url,
                                                                                    train_dataset_local_file_path))


# See Iris class


# noinspection PyShadowingNames
def parse_csv_line(line: str) -> tuple:
	"""
	Parse one line of a CSV file using tensorflow.
	:param line: The line of the CSV. It must contains five floats, separated by commas.
	:return: Return the features and the label of the line
	"""
	template = [[0.], [0.], [0.], [0.], [0]]
	parsed_line = tf.decode_csv(line, template)
	
	features = tf.reshape(parsed_line[:-1], shape=(4,))
	label = tf.reshape(parsed_line[-1], shape=())
	
	return features, label


# Load the CSV file (line by line)
training_dataset = tf.data.TextLineDataset(train_dataset_local_file_path)

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
"""
PRINT:
Example from a batch:
	features = tf.Tensor([5.1 3.8 1.9 0.4], shape=(4,), dtype=float32)
	label = tf.Tensor(0.0, shape=(), dtype=float32)
"""

"""
Selecting a model (as described earlier, 4 numbers (sepal length & width, petal length & width, and a label)
See https://www.tensorflow.org/images/custom_estimators/full_network.png to see the model
The model is a fully-connected Neural Network because it is very useful to find the relationship between the features
(sepal and petal) and the label.
In this example, keras will be used as a model selector.
Two fully-connected hidden layers containing 10 neurons each will be created. The output contains 3 neurons.
The activation function is ReLU : f(x) = {0 if x <= 0 ; x if x > 0}.
"""

model = tf.keras.Sequential([
	tf.keras.layers.Dense(units=10, activation="relu", input_shape=(4,)), # First hidden layers, with 4 for inputs
	tf.keras.layers.Dense(units=10, activation="relu"), # Second hidden layers, input automatically calculated
	tf.keras.layers.Dense(units=3)  # Output layer
])

### TRAINING ###

"""
Warning: if there is too much entries in the training set, then the model won't generalizable (see overfitting)
"""


# Creating a function to compute the error


# noinspection PyShadowingNames
def error(model: tf.keras.models.Model, x, y):
	"""
	Compute the error between the value returned by model(x) and the expected result y
	:param model: The model to use
	:param x: The input(s)
	:param y: The expected value that the model is supposed to return with x as input(s)
	:return: The error between the actual value and the expected value
	"""
	computed_y = model(x)
	return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=computed_y)


# noinspection PyShadowingNames
def grad(model: tf.keras.models.Model, inputs, targets):
	"""
	Record operations for backpropagation using the error() function.
	:param model:
	:param inputs:
	:param targets:
	:return:
	"""
	with tf.GradientTape() as tape:
		error_value = error(model, inputs, targets)
	return tape.gradient(error_value, model.variables)


"""
Now, the neural network needs an optimizer to minimize the error (using grad())
See the image https://tensorflow.org/images/opt1.gif for more detail
"""

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)  # This hyperparameter can be adjust for better result

"""
TRAINING LOOP
This is it! The neural network is ready to be fed by the dataset and learn!
"""

# Record results at each iteration

track_error = []
track_accuracy = []

max_iterations = 201

sw_training_loop = Stopwatch(start_now=True)
print("")
for iteration in range(max_iterations):
	iteration_error_average = tfe.metrics.Mean()
	iteration_accuracy = tfe.metrics.Accuracy()
	
	# Training loop
	for x, y in training_dataset:
		# Optimize the model
		grads = grad(model, x, y)
		optimizer.apply_gradients(zip(grads, model.variables),
		                          # See https://docs.python.org/3.3/library/functions.html#zip
		                          global_step=tf.train.get_or_create_global_step())
		
		# Track progress
		iteration_error_average(error(model, x, y))
		iteration_accuracy(tf.argmax(input=model(x), axis=1, output_type=tf.int32), y)
	
	# Stop iteration
	track_error.append(iteration_error_average.result())
	track_accuracy.append(iteration_accuracy.result())
	
	if iteration % 50 == 0:
		print("Iteration n°{0:03d}:\n\tError: {1:.3f}\n\tAccuracy: {2:.3%}\n".format(iteration, track_error[-1],
		                                                                             track_accuracy[-1]))
		"""
		PRINT:
		Iteration n°000:
			Error: 1.161
			Accuracy: 36.667%
		
		Iteration n°050:
			Error: 0.325
			Accuracy: 93.333%
		
		Iteration n°100:
			Error: 0.209
			Accuracy: 96.667%
		
		Iteration n°150:
			Error: 0.148
			Accuracy: 97.500%
		
		Iteration n°200:
			Error: 0.119
			Accuracy: 97.500%
		"""

sw_training_loop.stop()

print("Training time: {:0.2f}s".format(sw_training_loop.elapsed()))

### VIEW GRAPH ###
"""
Plot the errors and accuracy in diagrams
"""

# noinspection PyTypeChecker
fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(12, 8))
fig.suptitle("Training Metrics")

axes[0].set_ylabel("Error", fontsize=14)
axes[0].plot(track_error)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Iteration", fontsize=14)
axes[1].plot(track_accuracy)

#plt.show()

### TEST ###

test_dataset_url = "http://download.tensorflow.org/data/iris_test.csv"

test_dataset_local_file_path = tf.keras.utils.get_file(fname=os.path.basename(test_dataset_url),
                                                       origin=test_dataset_url)

print("A copy of the iris test dataset has been downloaded at from {0} to {1}.".format(test_dataset_url,
                                                                                       test_dataset_local_file_path))

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
assert test_accuracy.result() >= 0.8
"""
Test set accuracy: 100.000%
"""

### PREDICTIONS ###

predict_dataset = tf.convert_to_tensor([
	[5.1, 3.3, 1.7, 0.5],
	[5.9, 3.0, 4.2, 1.5],
	[6.9, 3.1, 5.4, 2.1]
])

# Pass the predict_dataset (without label) into the model (the neural network in this case)
predictions = model(predict_dataset)

print("Predictions:")
for i, logits in enumerate(predictions):
	class_idx = tf.argmax(logits).numpy()
	iris = Iris.from_int(class_idx)
	print("\tExample n°{}: Prediction: {} (logits = {})".format(i, Iris.to_str(iris), logits))
	
	if i == 0:
		assert iris == Iris.SETOSA
	elif i == 1:
		assert iris == Iris.VERSICOLOR
	elif i == 2:
		assert iris == Iris.VIRGINICA
	else:
		raise AssertionError("Iteration n°{} not expected".format(i))

"""
PRINT:
Predictions:
	Example n°0: Prediction: Iris setosa (logits = [  3.5572357    0.05438219 -10.779386  ])
	Example n°1: Prediction: Iris versicolor (logits = [-3.243308    1.7982299  -0.28065988])
	Example n°2: Prediction: Iris virginica (logits = [-5.9193463  2.2777824  4.0160313])
"""
