import torch
from primitives import funcprimitives
from stochastics import funcstochastics


env = {
        # deterministic functions
        '+': torch.add,
        '-': torch.subtract,
        '*': torch.multiply,
        '/': torch.divide,
        '=': torch.equal,
        '<': torch.lt,
        '>': torch.gt,
        '<=': torch.le,
        '>=': torch.ge,
        'sqrt': torch.sqrt,
        'mat-add': torch.add,
        'mat-mul': torch.matmul,
        'mat-tanh': torch.tanh,
        'mat-repmat': funcprimitives.repmat,
        'mat-transpose': funcprimitives.transpose,
        'if': funcprimitives.if_block,
        'vector': funcprimitives.vector,
        'first': funcprimitives.first,
        'last': funcprimitives.last,
        'append': funcprimitives.append,
        'get': funcprimitives.get,
        'hash-map': funcprimitives.hash_map,
        'put': funcprimitives.put,

        # stochastic functions
        'sample': funcstochastics.sample,
        'sample*': funcstochastics.sample,
        'discrete': funcstochastics.discrete,
        'uniform': funcstochastics.uniform,
        'normal': funcstochastics.normal,
        'beta': funcstochastics.beta,
        'exponential': funcstochastics.exponential
    }
