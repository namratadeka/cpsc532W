import torch

#TODO
class FuncPrimitives(object):
	def vector(self, *args):
		v = list()
		for arg in args:
			v.append(arg)

		try:
			return torch.Tensor(v)
		except:
			return v

	def first(self, tensor):
		return tensor[0]

	def last(self, tensor):
		return tensor[-1]

	def append(self, *args):
		vector = args[0]
		c = args[1]
		return torch.cat((torch.tensor(vector), torch.tensor([c])))

	def get(self, *args):
		iterator = args[0]
		key = args[1].item()

		if type(iterator) in [torch.Tensor, list]:
			key = int(key)
		return iterator[key]

	def put(self, *args):
		iterator = args[0]
		key = args[1].item()
		value = args[2]

		if type(iterator) == torch.Tensor:
			key = int(key)
		iterator[key] = value
		return iterator

	def hash_map(self, *args):
		assert (len(args) % 2 == 0)
		keys = list()
		values = list()
		for i in range(0, len(args)-1, 2):
			keys.append(args[i].item())
			values.append(args[i+1])
		return dict(zip(keys, values))

	def if_block(self, *args):
		predicate = args[0]
		if predicate:
			return args[1]
		return args[2]

	def transpose(self, arg):
		return torch.transpose(arg, 1, 0)

	def repmat(self, *args):
		x = args[0]
		dims = args[1:]
		return x.repeat(*dims)


funcprimitives = FuncPrimitives()
