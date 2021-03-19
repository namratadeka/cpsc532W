import torch
from tqdm import tqdm
from pyrsistent import pmap,plist

from daphne import daphne
from primitives import env as penv
from tests import is_tol, run_prob_test,load_truth


class Env(dict):
    def __init__(self, params=(), args=(), outer=None):
        self.update(zip(params, args))
        self.outer = outer
    def get(self, var):
        return self[var] if (var in self) else self.outer.get(var)

class Procedure(object):
    '''A user-defined function.'''
    def __init__(self, params, body, env):
        self.params, self.body, self.env = params, body, env
    def __call__(self, *args):
        return evaluate(self.body, Env(self.params, args, self.env))

def standard_env():
    "An environment with some Scheme standard procedures."
    env = pmap(penv)
    env = env.update({'alpha' : ''}) 

    return env


def evaluate(exp, env=None): #TODO: add sigma, or something
    if env is None:
        env = standard_env()

    if isinstance(exp, str):
        if env.get(exp) is not None:
            return env.get(exp)
        return exp
    elif not isinstance(exp, list):
        return torch.tensor(exp).float()
    op, *args = exp
    if op == 'if':
        (test, conseq, alt) = args
        exp = (conseq if evaluate(test, env) else alt)
        return evaluate(exp, env)
    elif op == 'sample':
        evaluate(args[0], env)
        dist = evaluate(args[1], env)
        return dist.sample()
    elif op == 'observe':
        evaluate(args[0], env)
        dist = evaluate(args[1], env)
        obs = evaluate(args[2], env)
        return obs
    elif op == 'fn':
        params, body = args
        return Procedure(params, body, env)
    else:
        proc = evaluate(op, env)
        vals = [evaluate(arg, env) for arg in args]
        return proc(*vals)

    return


def get_stream(exp):
    while True:
        yield evaluate(exp)('start-addr')


def run_deterministic_tests():
    
    for i in range(1,14):

        exp = daphne(['desugar-hoppl', '-i', '../CS532-HW5/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret = evaluate(exp)('start-addr')
        try:
            assert(is_tol(ret, truth))
        except:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,exp))
        
        print('FOPPL Tests passed')
        
    for i in range(1,13):

        exp = daphne(['desugar-hoppl', '-i', '../CS532-HW5/programs/tests/hoppl-deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/hoppl-deterministic/test_{}.truth'.format(i))
        ret = evaluate(exp)
        if isinstance(ret, Procedure):
            ret = ret('start-addr')
        try:
            assert(is_tol(ret, truth))
        except:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,exp))
        
        print('Test passed')
        
    print('All deterministic tests passed')
    


def run_probabilistic_tests():
    
    num_samples=1e4
    max_p_value = 1e-2
    
    for i in range(1,7):
        exp = daphne(['desugar-hoppl', '-i', '../CS532-HW5/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        stream = get_stream(exp)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        assert(p_val > max_p_value)
    
    print('All probabilistic tests passed')    



if __name__ == '__main__':
    
    run_deterministic_tests()
    run_probabilistic_tests()
    
    import sys
    sys.setrecursionlimit(10000)

    from plots import histogram, plot_heatmap

    for i in range(1,4):
        print(i)
        exp = daphne(['desugar-hoppl', '-i', '../CS532-HW5/programs/{}.daphne'.format(i)])
        print('\n\n\nSampling from prior of program {}:'.format(i))
        N, samples = 1e4, []
        stream = get_stream(exp)
        for j in tqdm(range(int(N))):
            samples.append(next(stream))

        samples = torch.stack(samples).float()
        samples = samples.reshape(int(N),-1)
        mean = samples.mean(dim=0)
        variance = samples.var(dim=0)
        print("Mean = {}\nVariance = {}".format(mean, variance))

        if i==3:
            plot_heatmap(mean, variance, name="Program {}".format(i+1))
        else:
            histogram(samples, name="Program {}".format(i+1))
