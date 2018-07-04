# -*- coding: utf-8 -*-

# See https://stackoverflow.com/questions/33759623/tensorflow-how-to-save-restore-a-model

import tensorflow as tf

from .eager_min import *

models_dir = "res/saved_models/eager"

with tf.Graph().as_default():
	with tf.Session() as sess:
		
		training_features = []
		training_labels = []
		
		for x, y in training_dataset:
			training_features.append(x)
			training_labels.append(y)
		
		inputs = {
			"batch_size": batch_size,
			"training_features": training_features,
			"training_labels": training_labels
		}
		
		outputs = {
			"prediction": training_labels
		}
		
		tf.saved_model.simple_save(
			session=sess,
			export_dir=models_dir,
			inputs=inputs,
			outputs=test_outputs
		)
		
		print("model saved")
