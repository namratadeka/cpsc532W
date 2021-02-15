import torch
import torch.distributions as dist

class FuncStochastics(object):
	def sample(self, *args):
		distribution = args[0]
		return distribution.sample()

	def normal(self, *args):
		loc = args[0]
		scale = args[1]
		return dist.normal.Normal(loc, scale)

	def beta(self, *args):
		return dist.beta.Beta(args[0], args[1])

	def exponential(self, *args):
		return dist.exponential.Exponential(rate=args[0])

	def uniform(self, *args):
		return dist.uniform.Uniform(low=args[0], high=args[1])


funcstochastics = FuncStochastics()