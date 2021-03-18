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
        

def push_addr(alpha, value):
    return str(alpha) + str(value)

def vector(addr, *args):
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

def conj(addr, data, el):
    if len(el.shape) == 0: el = el.reshape(1)
    return torch.cat([data, el], dim=0)

def hashmap(addr, *args):
    result, i = {}, 0
    while i<len(args):
        key, value  = args[i], args[i+1]
        if type(key) is torch.Tensor:
            key = key.item()
        result[key] = value
        i += 2
    return result

def get(addr, struct, index):
    if type(index) is torch.Tensor:
        index = index.item()
    if type(struct) in [torch.Tensor, list, tuple]:
        index = int(index)
    return struct[index]

def put(addr, struct, index, value):
    if type(index) is torch.Tensor:
        index = int(index.item())
    if type(struct) in [torch.Tensor, list, tuple]:
        index = int(index)
    result = deepcopy(struct)
    result[index] = value
    return result

eq = lambda addr, a, b: a==b

env = {
           'normal' : Normal,
           'push-address' : push_addr,
           '+': lambda addr, a, b: torch.add(a,b),
           '-': lambda addr, a, b: torch.subtract(a,b),
           '*': lambda addr, a, b: torch.multiply(a,b),
           '/': lambda addr, a, b: torch.divide(a,b),
           '>': lambda addr, a, b: a > b,
           '<': lambda addr, a, b: a < b,
           '=': eq,
           '==': eq,
           'or': lambda addr, a, b: a or b,
           'and': lambda addr, a, b: a and b,
           'sqrt': lambda addr, x: torch.sqrt(torch.tensor(x)),
           'first': lambda addr, data: data[0],
           'rest': lambda addr, data: data[1:],
           'last': lambda addr, data: data[-1],
           'vector': vector,
           'append': conj,
           'get': get,
           'hash-map': hashmap,
           'put': put,

       }






