import torch
import torch.distributions as dist
from copy import deepcopy



class Normal(dist.Normal):
    
    def __init__(self, alpha, loc, scale):
        
        if scale > 20.:
            self.optim_scale = scale.clone().detach().requires_grad_()
        else:
            self.optim_scale = torch.log(torch.exp(scale) - 1).clone().detach().requires_grad_()
        
        
        super().__init__(loc, torch.nn.functional.softplus(self.optim_scale))
    
    def Parameters(self):
        """Return a list of parameters for the distribution"""
        return [self.loc, self.optim_scale]
        
    def make_copy_with_grads(self):
        """
        Return a copy  of the distribution, with parameters that require_grad
        """
        
        ps = [p.clone().detach().requires_grad_() for p in self.Parameters()]
         
        return Normal(*ps)
    
    def log_prob(self, x):
        
        self.scale = torch.nn.functional.softplus(self.optim_scale)
        
        return super().log_prob(x)

def uniform(alpha, a, b):
    return dist.Uniform(a, b)

def dirichlet(alpha, concentration):
    return dist.Dirichlet(concentration)

def exponential(alpha, lamb):
    return dist.Exponential(lamb)

def discrete(alpha, vector):
    return dist.Categorical(vector)

def bernoulli(alpha, p, obs=None):
    return dist.Bernoulli(p)

def beta(alpha, concentration, rate, obs=None):
    return dist.Beta(concentration, rate)

def gamma(alpha, concentration, rate):
    return dist.Gamma(concentration, rate)

def push_addr(alpha, value):
    return alpha + value

def vector(alpha, *args):
    if len(args) == 0:
        return torch.tensor([])
    # sniff test: if what is inside isn't int,float,or tensor return normal list
    if type(args[0]) not in [int, float, torch.Tensor]:
        return [arg for arg in args]
    # if tensor dimensions are same, return stacked tensor
    if type(args[0]) is torch.Tensor:
        sizes = list(filter(lambda arg: arg.shape == args[0].shape, args))
        if len(sizes) == len(args):
            return torch.stack(args)
        else:
            return [arg for arg in args]
    raise Exception(f'Type of args {args} could not be recognized.')

def conj(alpha, data, el):
    if len(el.shape) == 0: el = el.reshape(1)
    return torch.cat([data, el], dim=0)

def cons(alpha, data, el):
    if len(el.shape) == 0: el = el.reshape(1)
    return torch.cat([el, data], dim=0)

def hashmap(alpha, *args):
    result, i = {}, 0
    while i<len(args):
        key, value  = args[i], args[i+1]
        if type(key) is torch.Tensor:
            key = key.item()
        result[key] = value
        i += 2
    return result

def get(alpha, struct, index):
    if type(index) is torch.Tensor:
        index = index.item()
    if type(struct) in [torch.Tensor, list, tuple]:
        index = int(index)
    return struct[index]

def put(alpha, struct, index, value):
    if type(index) is torch.Tensor:
        index = int(index.item())
    if type(struct) in [torch.Tensor, list, tuple]:
        index = int(index)
    result = deepcopy(struct)
    result[index] = value
    return result

eq = lambda alpha, a, b: a==b

env = {
           'normal' : Normal,
           'beta': beta,
           'gamma': gamma,
           'dirichlet': dirichlet,
           'uniform': uniform,
           'uniform-continuous': uniform,
           'exponential': exponential,
           'discrete': discrete,
           'flip': bernoulli,
           'bernoulli': bernoulli,

           'push-address' : push_addr,
           '+': lambda alpha, a, b: torch.add(a,b),
           '-': lambda alpha, a, b: torch.subtract(a,b),
           '*': lambda alpha, a, b: torch.multiply(a,b),
           '/': lambda alpha, a, b: torch.divide(a,b),
           '>': lambda alpha, a, b: a > b,
           '<': lambda alpha, a, b: a < b,
           '=': eq,
           '==': eq,
           'or': lambda alpha, a, b: a or b,
           'and': lambda alpha, a, b: a and b,
           'sqrt': lambda alpha, x: torch.sqrt(torch.tensor(x)),
           'first': lambda alpha, data: data[0],
           'rest': lambda alpha, data: data[1:],
           'last': lambda alpha, data: data[-1],
           'peek': lambda alpha, data: data[-1],
           'vector': vector,
           'append': conj,
           'conj': conj,
           'cons': cons,
           'get': get,
           'hash-map': hashmap,
           'put': put,
           'empty?': lambda alpha, a: len(a) == 0,
           'exp': lambda alpha, a: torch.exp(a),
           'log': lambda alpha, a: torch.log(a)
       }






