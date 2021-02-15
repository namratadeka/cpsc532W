import torch
from torch import tensor
from daphne import daphne
from tests import is_tol, run_prob_test,load_truth
from primitives import funcprimitives #TODO
from ops import env


lv = dict()
fn = dict()

def evaluate_defn(exp):
    fn[exp[0]] = {
                  'args': exp[1],
                  'body': exp[2]
                 }

def evaluate_let(exp):
    bindings = exp[0]
    ret_exp = exp[1]
    lv[bindings[0]] = bindings[1]
    return evaluate(ret_exp)


def evaluate(exp):
    "Evaluation function for the deterministic target language of the graph based representation."
    if type(exp) is list:
        op = exp[0]
        args = exp[1:]

        if op == 'let':
            return evaluate_let(args)

        if op == 'defn':
            return evaluate_defn(args)

        if op in fn:
            for i in range(len(fn[op]['args'])):
                lv[fn[op]['args'][i]] = args[i]
            return evaluate(fn[op]['body'])

        return env[op](*map(evaluate, args))
    elif type(exp) is str:
        if exp in lv:
            return evaluate(lv[exp])
        return exp
    elif type(exp) is int or type(exp) is float:
        # We use torch for all numerical objects in our evaluator
        return torch.tensor(float(exp))
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
        print(evaluate_program(ast)[0])