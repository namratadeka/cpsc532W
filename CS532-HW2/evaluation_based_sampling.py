import torch
from torch import tensor
from daphne import daphne
from tests import is_tol, run_prob_test,load_truth
from primitives import funcprimitives #TODO
from ops import env
import functools


fn = dict()
sigma = dict()

def evaluate_defn(exp):
    arg_values = [None for x in exp[1]]
    fn[exp[0]] = {
                  'args': dict(zip(exp[1], arg_values)),
                  'body': exp[2]
                 }

def evaluate_let(exp, lv={}):
    bindings = exp[0]
    ret_exp = exp[1]
    lv[bindings[0]] = bindings[1]
    return evaluate(ret_exp, lv=lv)

def evaluate(exp, lv={}):
    "Evaluation function for the deterministic target language of the graph based representation."
    if type(exp) is list:
        op = exp[0]
        args = exp[1:]

        if op == 'let':
            return evaluate_let(args, lv)

        if op == 'defn':
            return evaluate_defn(args)

        if op in fn:
            for i, key in enumerate(fn[op]['args']):
                fn[op]['args'][key] = evaluate(args[i],lv)
            return evaluate(fn[op]['body'], lv=fn[op]['args'].copy())

        if op in env:
            evaluate_bind = functools.partial(evaluate, lv=lv)
            return env[op](*map(evaluate_bind, args))

        return exp

    elif type(exp) is str:
        if exp in lv:
            return evaluate(lv[exp], lv)
        return exp

    elif type(exp) is int or type(exp) is float:
        # We use torch for all numerical objects in our evaluator
        return torch.tensor(float(exp))

    elif type(exp) is torch.Tensor:
        return exp

    else:
        raise("Expression type unknown.", exp)

        
def evaluate_program(ast):
    """Evaluate a program as desugared by daphne, generate a sample from the prior
    Args:
        ast: json FOPPL program
    Returns: sample from the prior of ast
    """
    # TODO
    for i in range(len(ast)):
        ret = evaluate(ast[i])
    return ret


def get_stream(ast):
    """Return a stream of prior samples"""
    while True:
        yield evaluate_program(ast)
    


def run_deterministic_tests():
    
    for i in range(1,14):
        #note: this path should be with respect to the daphne path!
        ast = daphne(['desugar', '-i', '../CS532-HW2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        print(ast)
        ret = evaluate_program(ast)
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,ast))
        
        print('Test passed')
        
    print('All deterministic tests passed')
    


def run_probabilistic_tests():

    num_samples=1e4
    max_p_value = 1e-4
    
    for i in range(1,7):
        #note: this path should be with respect to the daphne path!       
        ast = daphne(['desugar', '-i', '../CS532-HW2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        print(ast)
        stream = get_stream(ast)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        assert(p_val > max_p_value)
        print('Test passed')
    
    print('All probabilistic tests passed')    

        
if __name__ == '__main__':

    run_deterministic_tests()
    
    run_probabilistic_tests()


    for i in range(1,5):
        ast = daphne(['desugar', '-i', '../CS532-HW2/programs/{}.daphne'.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))
        ret = evaluate_program(ast)
        print(ret)

        fn.clear()
        sigma.clear()