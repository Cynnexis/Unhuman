# -*- coding: utf-8 *-*
# Code duplication from eager.py, but minimize

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.contrib.eager as tfe


class EagerClass:
	
	train_dataset_url = "http://download.tensorflow.org/data/iris_training.csv"
	test_dataset_url = "http://download.tensorflow.org/data/iris_test.csv"
	batch_size = 32
	
	def __init__(self):
		"""
		Initialize the model
		"""
		if not tf.executing_eagerly():
			tf.enable_eager_execution()

		self.train_dataset_local_file_path = tf.keras.utils.get_file(fname=os.path.basename(self.train_dataset_url),
		                                                        origin=self.train_dataset_url)
		self.test_dataset_local_file_path  = tf.keras.utils.get_file(fname=os.path.basename(self.test_dataset_url),
		                                                        origin=self.test_dataset_url)
		
		self.training_dataset = None
		self.model = None
		self.optimizer = None
		self.test_dataset = None
		self.test_accuracy = None
		self.test_outputs = None
	
	def train(self, training_dataset_path=None, display_every_n_iteration: int = 100) -> tuple:
		"""
		Train the model
		:param training_dataset_path: The path to the training data set file
		:param display_every_n_iteration: Display the loss and accuracy of the model each 'n' iteration. Set it to a
		value less or equal to 0 to disable it
		:return: Return a tuple containing the model as first element, and the optimizer as second element
		"""
		def grad(model: tf.keras.models.Model, inputs, targets):
			with tf.GradientTape() as tape:
				error_value = tf.losses.sparse_softmax_cross_entropy(labels=targets, logits=model(inputs))
			return tape.gradient(error_value, model.variables)
		
		if training_dataset_path is None:
			training_dataset_path = self.train_dataset_local_file_path
		
		self.training_dataset = tf.data.TextLineDataset(training_dataset_path)
		self.training_dataset = self.training_dataset.skip(1)
		self.training_dataset = self.training_dataset.map(self.parse_csv_line)
		self.training_dataset = self.training_dataset.shuffle(buffer_size=1000)
		self.training_dataset = self.training_dataset.batch(batch_size=self.batch_size)
		
		self.model = tf.keras.Sequential([
			tf.keras.layers.Dense(units=10, activation="relu", input_shape=(4,)),
			tf.keras.layers.Dense(units=10, activation="relu"),
			tf.keras.layers.Dense(units=3)
		])
		
		self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
		
		track_error = []
		track_accuracy = []
		
		for iteration in range(200 + 1):
			iteration_error_average = tfe.metrics.Mean()
			iteration_accuracy = tfe.metrics.Accuracy()
			
			for x, y in self.training_dataset:
				grads = grad(self.model, x, y)
				self.optimizer.apply_gradients(grads_and_vars=zip(grads, self.model.variables),
				                          global_step=tf.train.get_or_create_global_step(),
				                          name="optimizer_log")
				
				if display_every_n_iteration > 0:
					iteration_error_average(self.error(self.model, x, y))
					iteration_accuracy(tf.argmax(input=self.model(x), axis=1, output_type=tf.int32), y)
			
			if display_every_n_iteration > 0:
				track_error.append(iteration_error_average.result())
				track_accuracy.append(iteration_accuracy.result())
			
			if display_every_n_iteration > 0 and iteration % display_every_n_iteration == 0:
				print("Iteration n°{0:03d}:\n\tError: {1:.3f}\n\tAccuracy: {2:.3%}\n".format(iteration, track_error[-1],
				                                                                             track_accuracy[-1]))
			
			# TensorBoard (see eager_tensorboard.py)
			tf.summary.tensor_summary(
				name="Error",
				tensor=tf.convert_to_tensor(value=track_error)
			)
			
			tf.summary.tensor_summary(
				name="Accuracy",
				tensor=tf.convert_to_tensor(value=track_accuracy)
			)
		
		return self.model, self.optimizer
	
	def test(self, test_dataset_path=None, display: bool = True) -> float:
		"""
		Test the model
		:param test_dataset_path: The path to the test data set file
		:param display: If True, the accuracy will be shown as a percentage
		:return: The accuracy of the model
		"""
		if test_dataset_path is None:
			test_dataset_path = self.test_dataset_local_file_path
		
		self.test_dataset = tf.data.TextLineDataset(test_dataset_path)
		self.test_dataset = self.test_dataset.skip(1)
		self.test_dataset = self.test_dataset.map(self.parse_csv_line)
		self.test_dataset = self.test_dataset.shuffle(1000)
		self.test_dataset = self.test_dataset.batch(self.batch_size)
		
		self.test_accuracy = tfe.metrics.Accuracy()
		
		self.test_outputs = []
		
		for x, y in self.test_dataset:
			self.test_outputs.append(self.model(x))
			prediction = tf.argmax(self.test_outputs[-1], axis=1, output_type=tf.int32)
			self.test_accuracy(prediction, y)
		
		if display:
			print("Test set accuracy: {:.3%}".format(self.test_accuracy.result()))
		
		return self.test_accuracy.result()
	
	def predict(self, inputs):
		"""
		Make a prediction according to the inputs.
		:param inputs: The inputs of the model
		:return: Return the prediction by the model
		"""
		return self.model(inputs)
	
	@staticmethod
	def main(keep_training_until_threshold: bool = True, threshold: float = .9, training_dataset_path=None,
	         test_dataset_path=None):
		"""
		Construct an EagerClass instance, train it and test it.
		:param keep_training_until_threshold: If True, the method will construct, train and test again the model until
		the accuracy given by the test is greater than the threshold. If False, construct one instance, train it and
		test it regardless of the accuracy
		:param threshold: The accuracy threshold
		:param training_dataset_path: The path to the training data set file
		:param test_dataset_path: The path to the test data set file
		:return: Return a model fully trained and tested
		"""
		if not tf.executing_eagerly():
			tf.enable_eager_execution()
		
		accuracy = 0.
		em = None
		
		if keep_training_until_threshold:
			while accuracy < threshold:
				em = EagerClass()
				em.train(training_dataset_path)
				accuracy = em.test(test_dataset_path)
		else:
			em = EagerClass()
			em.train(training_dataset_path)
			em.test(test_dataset_path)
		
		return em
	
	@staticmethod
	def parse_csv_line(line: str) -> tuple:
		"""
		Parse one line in a CSV file
		:param line: The current line
		:return: The elements separated
		"""
		template = [[0.], [0.], [0.], [0.], [0]]
		parsed_line = tf.decode_csv(line, template)
		features = tf.reshape(parsed_line[:-1], shape=(4,))
		label = tf.reshape(parsed_line[-1], shape=())
		return features, label
	
	@staticmethod
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

