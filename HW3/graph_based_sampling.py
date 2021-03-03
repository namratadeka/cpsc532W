import torch
import torch.distributions as dist
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from copy import deepcopy

from daphne import daphne
import numpy as np

from primitives import PRIMITIVES
from tests import is_tol, run_prob_test,load_truth
from evaluation_based_sampling import evaluate_program

# Put all function mappings from the deterministic language environment to your
# Python evaluation context here:
# env = {'normal': dist.Normal,
#        'sqrt': torch.sqrt}
env = PRIMITIVES

def deterministic_eval(exp):
    "Evaluation function for the deterministic target language of the graph based representation."
    if type(exp) is list:
        op = exp[0]
        args = exp[1:]
        return env[op](*map(deterministic_eval, args))
    elif type(exp) in [int, float, bool]:
        # We use torch for all numerical objects in our evaluator
        return torch.tensor([float(exp)], requires_grad=True).squeeze()
    elif type(exp) is torch.Tensor:
        return exp
    else:
        raise Exception("Expression type unknown.", exp)

def topological_sort(nodes, edges):
    result = []
    visited = {}
    def helper(node):
        if node not in visited:
            visited[node] = True
            if node in edges:
                for child in edges[node]:
                    helper(child)
            result.append(node)
    for node in nodes:
        helper(node)
    return result[::-1]

def plugin_parent_values(expr, trace):
    if type(expr) == str and expr in trace:
        return trace[expr]
    elif type(expr) == list:
        return [plugin_parent_values(child_expr, trace) for child_expr in expr]
    else:
        return expr

def sample_from_joint(graph):
    "This function does ancestral sampling starting from the prior."
    # TODO insert your code here
    """
    1. Run topological sort on V using V and A, resulting in an array of v's
    2. Iterate through sample sites of the sorted array, and save sampled results on trace dictionary using P and Y
    - If keyword is sample*, first recursively replace sample site names with trace values in the expression from P. Then, run deterministic_eval.
    - If keyword is observe*, put the observation value in the trace dictionary
    3. Filter the trace dictionary for things sample sites you should return
    """
    procs, model, expr = graph[0], graph[1], graph[2]
    nodes, edges, links, obs = model['V'], model['A'], model['P'], model['Y']
    sorted_nodes = topological_sort(nodes, edges)

    sigma = {'logW':0}
    trace = {}
    for node in sorted_nodes:
        keyword = links[node][0]
        if keyword == "sample*":
            link_expr = links[node][1]
            link_expr = plugin_parent_values(link_expr, trace)
            dist_obj  = deterministic_eval(link_expr)
            trace[node] = dist_obj.sample()
        elif keyword == "observe*":
            trace[node] = obs[node]
            link_expr = links[node][1]
            link_expr = plugin_parent_values(link_expr, trace)
            dist_obj  = deterministic_eval(link_expr)
            sigma['logW'] += dist_obj.log_prob(obs[node])

    expr = plugin_parent_values(expr, trace)
    return deterministic_eval(expr), sigma, trace


def get_stream(graph):
    """Return a stream of prior samples
    Args: 
        graph: json graph as loaded by daphne wrapper
    Returns: a python iterator with an infinite stream of samples
        """
    while True:
        yield sample_from_joint(graph)




#Testing:

def run_deterministic_tests():
    
    for i in range(1,13):
        #note: this path should be with respect to the daphne path!
        graph = daphne(['graph','-i','../CS532-HW2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret = deterministic_eval(graph[-1])
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for graph {}'.format(ret,truth,graph))
        
        print('Test passed')
        
    print('All deterministic tests passed')



def run_probabilistic_tests():
    
    #TODO: 
    num_samples = 1e4
    max_p_value = 1e-4
    
    for i in range(1,7):
        #note: this path should be with respect to the daphne path!        
        graph = daphne(['graph', '-i', '../CS532-HW2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        stream = get_stream(graph)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        print(p_val > max_p_value)
    
    print('All probabilistic tests passed')    


def print_tensor(tensor):
    tensor = np.round(tensor.numpy(), decimals=3)
    print(tensor)
        

def hw_2():
    run_deterministic_tests()
    run_probabilistic_tests()

    for i in range(1,5):
        graph = daphne(['graph','-i','../CS532-HW2/programs/{}.daphne'.format(i)])
        samples, n = [], 1000
        for j in range(n):
            sample = sample_from_joint(graph)[0]
            samples.append(sample)

        print("\nExpectation of return values for program {i}:")
        if type(samples[0]) is list:
            expectation = [None]*len(samples[0])
            for j in range(n):
                for k in range(len(expectation)):
                    if expectation[k] is None:
                        expectation[k] = [samples[j][k]]
                    else:
                        expectation[k].append(samples[j][k])
            for k in range(len(expectation)):
                print_tensor(sum(expectation[k])/n)
        else:
            expectation = sum(samples)/n
            print_tensor(expectation)

def accept(x, X_, X, edges, links, obs):
    q = deterministic_eval(plugin_parent_values(links[x][1], {**X, **obs}))
    q_ = deterministic_eval(plugin_parent_values(links[x][1], {**X_, **obs}))
    log_alpha = q_.log_prob(X_[x]) - q.log_prob(X[x])
    Vx = edges[x]
    for v in Vx:
        link_expr = plugin_parent_values(links[v][1], {**X_, **obs})
        dist_obj = deterministic_eval(link_expr)
        log_alpha += dist_obj.log_prob({**X_, **obs}[v])

        link_expr = plugin_parent_values(links[v][1], {**X, **obs})
        dist_obj = deterministic_eval(link_expr)
        log_alpha -= dist_obj.log_prob({**X_, **obs}[v])

    return torch.exp(log_alpha)


def gibbs_step(graph, trace):
    procs, model, expr = graph[0], graph[1], graph[2]
    nodes, edges, links, obs = model['V'], model['A'], model['P'], model['Y']
    sorted_nodes = topological_sort(nodes, edges)
    
    diff = list(sorted_nodes - obs.keys())
    X = [node for node in sorted_nodes if node in diff]
    X_dict = {}
    for x in X:
        X_dict[x] = trace[x]
    X = X_dict

    for x in X:
        link_expr = plugin_parent_values(links[x][1], {**X, **obs})
        q = deterministic_eval(link_expr)
        X_ = X.copy()
        X_[x] = q.sample()
        alpha = accept(x, X_, X, edges, links, obs)
        u = torch.distributions.Uniform(0,1).sample()
        if (u < alpha).all():
            X = X_

    return_trace = {**X, **obs}
    return_expr = plugin_parent_values(expr, return_trace)
    return deterministic_eval(return_expr), return_trace


def hw_3_gibbs():
    for i in range(1,5):
        graph = daphne(['graph', '-i', '../HW3/hw3-programs/{}.daphne'.format(i)])
        samples, n = [], 10000
        _, _, trace = sample_from_joint(graph)
        for j in tqdm(range(n)):
            sample, trace = gibbs_step(graph, trace)
            samples.append(sample)

        samples = torch.stack(samples).float()
        mean = samples.mean(dim=0)
        var = samples.var(dim=0)

        print("Posterior mean for program-{}: {}".format(i, mean))
        print("Posterior variance for program-{}: {}".format(i, var))


def tensor_dict_add(x, r):
    y = {}
    for i, key in enumerate([*x.keys()]):
        y[key] = x[key].detach() + r[i]
        y[key].requires_grad = True
    return y

def H(X, R, M, obs, links):
    return U(X, obs, links) + 0.5*torch.matmul(R, torch.matmul(M.inverse(), R))

def U(X, obs, links):
    logP = 0
    for node in obs:
        link_expr = plugin_parent_values(links[node][1], {**X, **obs})
        dist_obj = deterministic_eval(link_expr)
        logP += dist_obj.log_prob(obs[node])

    return -logP

def del_U(X, obs, links):
    Ux = U(X, obs, links)
    Ux.backward()
    gradients = torch.zeros(len(X))
    for i, key in enumerate([*X.keys()]):
        gradients[i] = X[key].grad

    return gradients

def leapfrog(X0, R0, T, epsilon, obs, links):
    X = {0:X0}
    R = {0:R0}
    R[1/2] = R[0] - 0.5*epsilon*del_U(X0, obs, links)
    for t in range(1, T):
        X[t] = tensor_dict_add(X[t-1], epsilon*R[t - 1/2])
        R[t + 1/2] = R[t - 1/2] - epsilon*del_U(X[t], obs, links)
    X[T] = tensor_dict_add(X[t-1], epsilon*R[T - 1/2])
    R[T] = R[T - 1/2] - 0.5*epsilon*del_U(X[T], obs, links)

    return X[T], R[T]

def hmc(X, S, T, epsilon, M, obs, links):
    traces = []
    r_dist = torch.distributions.MultivariateNormal(torch.zeros(len(X)), M)
    for s in tqdm(range(S)):
        R = r_dist.sample()
        X_, R_ = leapfrog(deepcopy(X), R, T, epsilon, obs, links)
        u = torch.distributions.Uniform(0, 1).sample()
        if u < torch.exp(-H(X_, R_, M, obs, links) + H(X, R, M, obs, links)):
            X = X_
        traces.append(X)

    return traces

def hw3_hmc():
    for i in [1,2]:
        graph = daphne(['graph', '-i', '../HW3/hw3-programs/{}.daphne'.format(i)])
        procs, model, expr = graph[0], graph[1], graph[2]
        nodes, edges, links, obs = model['V'], model['A'], model['P'], model['Y']
        sorted_nodes = topological_sort(nodes, edges)
        diff = list(sorted_nodes - obs.keys())
        X = [node for node in sorted_nodes if node in diff]

        samples = []
        _, _, trace = sample_from_joint(graph)
        X_dict = {}
        for x in X:
            X_dict[x] = trace[x]
            X_dict[x].requires_grad = True
        X = X_dict

        M = torch.eye(len(X))
        S = 10000
        epsilon = 0.1
        T = 10

        X_traces = hmc(X, S, T, epsilon, M, obs, links)
        
        for x in X_traces:
            final_trace = {**x, **obs}
            final_expr = plugin_parent_values(expr, final_trace)
            samples.append(deterministic_eval(final_expr))

        samples = torch.stack(samples).float()
        mean = samples.mean(dim=0)
        var = samples.var(dim=0)

        print("Posterior mean for program-{}: {}".format(i, mean))
        print("Posterior variance for program-{}: {}\n".format(i, var))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Prob. Prog. Assignments", allow_abbrev=False
    )
    parser.add_argument(
        "-s",
        "--sampler",
        type=str,
        required=True,
    )
    (args, unknown_args) = parser.parse_known_args()
    if args.sampler == 'hw2':
        hw_2()
    if args.sampler == 'hw3_gibbs':
        hw_3_gibbs()
    if args.sampler == 'hw3_hmc':
        hw3_hmc()
