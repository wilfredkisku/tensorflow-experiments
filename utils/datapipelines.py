import tensorflow as tf

import pathlib
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4)

class Pipelines:

	def createDataset():

		dataset = tf.data.Dataset.from_tensor_slices([8, 3, 0, 8, 2, 1])
		print(dataset)
		
		#print directly
		for element in dataset:
			print(element.numpy())

		#print using an iterator
		it = iter(dataset)
		for x in it:
			print(x.numpy())
		
		print(dataset.reduce(0, lambda state, value: state + value).numpy())

		dataset1 = tf.data.Dataset.from_tensor_slices(tf.random.uniform([4, 10]))
		dataset2 = tf.data.Dataset.from_tensor_slices((tf.random.uniform([4]), tf.random.uniform([4, 10], maxval = 100, dtype=tf.int32)))
		dataset3 = tf.data.Dataset.zip((dataset1, dataset2))

		dataset4 = tf.data.Dataset.from_tensors(tf.SparseTensor(indices=[[0,0],[1,2]], values=[1,2], dense_shape=[3,4]))
	
		for a, (b, c) in dataset3:
			print('shapes: {x.shape}, {y.shape}, {z.shape}'.format(x=a,y=b,z=c))
		return None

	
	def inputData():
		train, test = tf.keras.datasets.fashion_mnist.load_data()
		images, labels = train
		images = images/255

		dataset = tf.data.Dataset.from_tensor_slices((images, labels))
		print(dataset)
		
		ds_counter = tf.data.Dataset.from_generator(count, args=[25], output_types=tf.int32, output_shapes = (),)

		return None

	def gen_series():
		i = 0
		while True:
			size =np.random.randint(0, 10)
			yield i, np.random.normal(size=(size,))
			i += 1

	def imagePipelines():
		
		flowers_path = tf.keras.utils.get_file('flower_photos','https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',untar=True)
		img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=20)
		images, labels = next(img_gen.flow_from_directory(flowers_path))
		print(images.dtype, images.shape)
		print(labels.dtype, labels.shape)

		ds = tf.data.Dataset.from_generator(lambda: img_gen.flow_from_directory(flowers_path), output_types=(tf.float32, tf.float32), output_shapes=([32, 256, 256, 3],[32, 5]))
		print(ds.element_spec)
if __name__ == "__main__":
	
	#Pipelines.createDataset()
	#Pipelines.inputData()
	#for i, series in Pipelines.gen_series():
	#	print(i, ":" , str(series))
	#	if i > 5:
	#		break
	#ds_series = tf.data.Dataset.from_generator(Pipelines.gen_series, output_types=(tf.int32, tf.float32), output_shapes=((), (None,)))

	#ds_series_batch = ds_series.shuffle(20).padded_batch(10)
	#ids, sequence_batch = next(iter(ds_series_batch))
	#print(ids.numpy())
	#print(sequence_batch.numpy())
	Pipelines.imagePipelines()
