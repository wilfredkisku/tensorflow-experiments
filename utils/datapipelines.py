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
if __name__ == "__main__":
	
	Pipelines.createDataset()
