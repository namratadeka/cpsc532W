import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from collections import namedtuple

from daphne import daphne
from primitives import PRIMITIVES
from eval_utils import deterministic_eval, topological_sort, plugin_parent_values


env = PRIMITIVES


def grad_log_prob(dist, c):
    dg = dist.make_copy_with_grads()
    logQ = dg.log_prob(c)
    logQ.backward()
    grads = []
    for i in range(len(dg.Parameters())):
        grads.append(dg.Parameters()[i].grad)

    return torch.stack(grads)

def sample_from_joint(graph, sigma):
    procs, model, expr = graph[0], graph[1], graph[2]
    nodes, edges, links, obs = model['V'], model['A'], model['P'], model['Y']
    sorted_nodes = topological_sort(nodes, edges)

    trace = {}
    loc_sigma = {'logW': 0, 'q':sigma['q'], 'G':{}, 'opt': sigma['opt']}
    for node in sorted_nodes:
        keyword = links[node][0]
        if keyword == "sample*":
            link_expr = links[node][1]
            link_expr = plugin_parent_values(link_expr, trace)
            dist_obj  = deterministic_eval(link_expr)
            if node not in sigma['q']:
                loc_sigma['q'][node] = dist_obj.make_copy_with_grads()
            trace[node] = loc_sigma['q'][node].sample()
            loc_sigma['G'][node] = grad_log_prob(loc_sigma['q'][node], trace[node])
            loc_sigma['logW'] += dist_obj.log_prob(trace[node]) - loc_sigma['q'][node].log_prob(trace[node])
        elif keyword == "observe*":
            trace[node] = obs[node]
            link_expr = links[node][1]
            link_expr = plugin_parent_values(link_expr, trace)
            dist_obj  = deterministic_eval(link_expr)
            loc_sigma['logW'] += dist_obj.log_prob(obs[node])

    expr = plugin_parent_values(expr, trace)
    return deterministic_eval(expr), loc_sigma, trace


def covar(Fv, G, v):
    Fv = torch.stack(Fv)
    Gv = []
    for g in G:
        Gv.append(g[v])
    Gv = torch.stack(Gv)
    cov_FG = torch.matmul(Fv.squeeze().t(), Gv.squeeze()) - torch.matmul(Fv.squeeze().mean(dim=0).t(), Gv.squeeze().mean(dim=0))
    return cov_FG, Fv, Gv

def optimizer_step(Q, g_hat, opt):
    for v in g_hat:
        Q[v] = Q[v].make_copy_with_grads()
        for i in range(len(Q[v].Parameters())):
            Q[v].Parameters()[i].grad = -g_hat[v][i].detach()
        optim = torch.optim.SGD(Q[v].Parameters(), lr=1e-4)
        optim.step()
        optim.zero_grad()

    return Q

def elbo_gradients(G, logW):
    g_hat = {}
    dom = list()
    for g in G:
        dom += [*g.keys()]
    dom = list(set(dom))
    F = {}
    for v in dom:
        F[v] = list()
        for l in range(len(G)):
            if v in G[l]:
                F[v].append(G[l][v] * logW[l])
            else:
                F[v].append(0)
                G[l][v] = 0

        Fv = torch.stack(F[v])
        cov_FG, Fv, Gv = covar(F[v], G, v)
        b_hat = (cov_FG.sum() - cov_FG.trace()) / Gv.var(dim=0).sum()
        g_hat[v] = (Fv - b_hat*Gv).mean(dim=0)

    return g_hat

def bbvi_b(S, L, graph):
    sigma = {'logW':0, 'q':{}, 'G':{}, 'opt':{}}
    r = [[list() for l in range(L)] for s in range(S)]
    G = [[list() for l in range(L)] for s in range(S)]
    logW = [[list() for l in range(L)] for t in range(S)]
    iterator = tqdm(range(S))
    for s in range(S):
        for l in range(L):
            r[s][l], sigma_, trace = sample_from_joint(graph, sigma)
            G[s][l], logW[s][l] = sigma_['G'], sigma_['logW']
        g_hat = elbo_gradients(G[s], logW[s])
        sigma['q'] = optimizer_step(sigma_['q'], g_hat, sigma_['opt'])

        print("{}".format(sigma['q']))

    return r, sigma

def elbo(L, sigma, graph):
    logQ = []
    logP = []
    procs, model, expr = graph[0], graph[1], graph[2]
    nodes, edges, links, obs = model['V'], model['A'], model['P'], model['Y']
    for l in range(L):
        trace = {}
        logq = 0
        for v in sigma['q']:
            trace[v] = sigma['q'][v].sample()
            logq += sigma['q'][v].log_prob(trace[v])
        logQ.append(logq)

        logp = 0
        for node in obs:
            link_expr = plugin_parent_values(links[node][1], {**trace, **obs})
            dist_obj = deterministic_eval(link_expr)
            logp += dist_obj.log_prob(obs[node])
        logP.append(logp)

    logP = torch.stack(logP)
    logQ = torch.stack(logQ)

    return (logP - logQ).mean()

def bbvi(S, L, graph):
    sigma = {'logW':0, 'q':{}, 'G':{}, 'opt':{}}
    sample, sigma, trace = sample_from_joint(graph, sigma)
    sigma['opt'] = {}
    for node in sigma['q']:
        sigma['opt'][node] = torch.optim.SGD(sigma['q'][node].Parameters(), lr=1e-2)

    iterator = tqdm(range(S))
    for s in range(S):
        objective = -1 * elbo(L, sigma, graph)
        objective.backward()
        for node in sigma['q']:
            sigma['opt'][node].step()
            sigma['opt'][node].zero_grad()

        print("{}".format(sigma['q']))

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="BBVI", allow_abbrev=False
    )
    parser.add_argument(
        "-c",
        "--control",
        action="store_true",
        help="use control variate"
    )
    (args, unknown_args) = parser.parse_known_args()
    for i in range(4,5):
        graph = daphne(['graph','-i','../CS532-HW4/programs/{}.daphne'.format(i)])
        S, L = 10000, 1000
        if args.control:
            r, sigma = bbvi_b(S, L, graph)
        else:
            bbvi(S, L, graph)

if __name__=="__main__":
    main()
