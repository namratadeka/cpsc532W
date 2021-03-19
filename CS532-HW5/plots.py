import torch
import numpy as np
from seaborn import heatmap
import matplotlib.pyplot as plt


def histogram(samples, weights=None, name=None):
	n = len(samples)
	samples = samples.reshape(n, -1).detach().cpu().numpy()
	if weights is not None:
		weights = weights.detach().cpu().numpy()
	d = samples.shape[1]
	for i in range(d):
		plt.figure()
		plt.hist(samples[:, i], weights=weights, bins=100)
		plt.title("{}, dimension: {}".format(name, i))
		plt.show()

	# for i in range(d):
	# 	plt.plot(list(range(n)), samples[:,i])
	# 	plt.xlabel("#sample")
	# 	plt.title("Sample Trace for {}, dimension: {}".format(name, i))
	# 	plt.show()

def plot_heatmap(mean, variance, name=None):
	mean = mean.reshape(-1,1).detach().cpu().numpy()
	variance = variance.reshape(-1,1).detach().cpu().numpy()

	plt.figure(figsize=(20, 10))
	plt.subplot(1, 2, 1)
	heatmap(mean, annot=True)
	plt.title("Mean heatmap for each output dimension of {}.".format(name))

	plt.subplot(1, 2, 2)
	heatmap(variance, annot=True)
	plt.title("Variance heatmap for each output dimension of {}.".format(name))
	
	plt.show()