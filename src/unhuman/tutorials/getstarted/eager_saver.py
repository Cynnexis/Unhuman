# -*- coding: utf-8 -*-

# See https://stackoverflow.com/questions/33759623/tensorflow-how-to-save-restore-a-model

import tensorflow as tf
tf.enable_eager_execution()

from unhuman.tutorials.getstarted.eager_class import EagerClass

em = EagerClass.main()

models_dir = "res/saved_models/eager"

g = tf.Graph()
with g.as_default():
	with tf.Session() as sess:
		
		training_features = []
		training_labels = []
		
		for x, y in em.training_dataset:
			training_features.append(x)
			training_labels.append(y)
		
		inputs = {
			"batch_size": em.batch_size,
			"training_features": training_features,
			"training_labels": training_labels
		}
		
		outputs = {
			"prediction": training_labels[-1]
		}
		
		print("DEBUG> inputs (" + str(type(inputs)) + ") = " + str(inputs) + "\n")
		print("DEBUG> outputs (" + str(type(outputs)) + ") = " + str(outputs) + "\n")
		
		tf.saved_model.simple_save(
			session=sess,
			export_dir=models_dir,
			inputs=inputs,
			outputs=outputs
		)
		
		print("model saved")
