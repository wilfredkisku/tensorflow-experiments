import tensorflow as tf

import time
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


class ArtificialDataset():

	def _generator(num_samples):
		# opening the file
		time.sleep(0.03)
		
		for sample_idx in range(num_samples):
			# reading the (line, record) from the file
			time.sleep(0.015)
			yield (sample_idx,)


	def __new__(cls, num_samples=3):

		return tf.data.Dataset.from_generator(cls._generator, output_types=tf.dtypes.int64, output_shapes=(1,), args=(num_samples,))
	
	def benchmark(dataset, num_epochs = 2):
		start_time = time.perf_counter()
		for epoch_num in range(num_epochs):
			for sample in dataset:
				time.sleep(0.01)
		tf.print("Executin Time:", time.perf_counter() - start_time)

	def fast_benchmark(dataset, num_epochs=2):
		start_time = time.perf_counter()
		for _ in tf.data.Dataset.range(num_epochs):
			for _ in dataset:
				pass
		tf.print("Execution time:", time.perf_counter() - start_time)

	def increment(x):
		return x+1
		
if __name__ == "__main__":
	
	##### Working with the tensorflow dataset ocjects #####
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
	#Pipelines.imagePipelines()
	
	##### data pipelines #####
	ArtificialDataset.benchmark(ArtificialDataset())
	ArtificialDataset.benchmark(ArtificialDataset().prefetch(tf.data.experimental.AUTOTUNE))
	ArtificialDataset.benchmark(tf.data.Dataset.range(2).interleave(ArtificialDataset))
	ArtificialDataset.benchmark(tf.data.Dataset.range(2).interleave(ArtificialDataset, num_parallel_calls = tf.data.experimental.AUTOTUNE))
	fast_dataset = tf.data.Dataset.range(10000)
	ArtificialDataset.fast_benchmark(fast_dataset.map(ArtificialDataset.increment).batch(256))
	ArtificialDataset.fast_benchmark(fast_dataset.batch(256).map(ArtificialDataset.increment))
