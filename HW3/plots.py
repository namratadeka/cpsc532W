import torch
import numpy as np
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

	for i in range(d):
		plt.plot(list(range(n)), samples[:,i])
		plt.xlabel("#sample")
		plt.title("Sample Trace for {}, dimension: {}".format(name, i))
		plt.show()
