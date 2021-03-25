from evaluator import evaluate
import torch
import numpy as np
import json
import sys






def run_until_observe_or_end(res):
    cont, args, sigma = res
    res = cont(*args)
    while type(res) is tuple:
        if res[2]['type'] == 'observe':
            return res
        cont, args, sigma = res
        res = cont(*args)

    res = (res, None, {'done' : True}) #wrap it back up in a tuple, that has "done" in the sigma map
    return res

def resample_particles(particles, log_weights):
    #TODO
    new_particles = []

    weights = torch.exp(torch.FloatTensor(log_weights))
    normalized_weights = weights / weights.sum()
    logZ = torch.log(normalized_weights.mean())

    particle_indices = torch.multinomial(normalized_weights, len(particles), True)
    for idx in particle_indices:
        new_particles.append(particles[idx])

    return logZ, new_particles



def SMC(n_particles, exp):

    particles = []
    weights = []
    logZs = []
    output = lambda x: x

    for i in range(n_particles):

        res = evaluate(exp, env=None)('addr_start', output)
        logW = 0.


        particles.append(res)
        weights.append(logW)

    #can't be done after the first step, under the address transform, so this should be fine:
    done = False
    smc_cnter = 0
    while not done:
        current_addr = ''
        print('In SMC step {}, Zs: '.format(smc_cnter), logZs)
        for i in range(n_particles): #Even though this can be parallelized, we run it serially
            res = run_until_observe_or_end(particles[i])
            if 'done' in res[2]: #this checks if the calculation is done
                particles[i] = res[0]
                if i == 0:
                    done = True  #and enforces everything to be the same as the first particle
                    address = ''
                else:
                    if not done:
                        raise RuntimeError('Failed SMC, finished one calculation before the other')
            else:
                #TODO: check particle addresses, and get weights and continuations
                particles[i] = res
                if i == 0:
                    current_addr = res[2]['addr']
                else:
                    addr = res[2]['addr']
                    if not current_addr == addr:
                        raise RuntimeError('Failed SMC, address mismatch. Expected {} but got {}.'.format(current_addr, addr))
                logW = res[2]['logW']
                weights[i] += logW

        if not done:
            #resample and keep track of logZs
            logZn, particles = resample_particles(particles, weights)
            logZs.append(logZn)
            weights = [0.] * len(weights) # reset weights
        smc_cnter += 1
    logZ = sum(logZs)
    return logZ, particles


if __name__ == '__main__':

    for i in range(1,5):
        with open('programs/{}.json'.format(i),'r') as f:
            exp = json.load(f)
        n_particles = [int(10**x) for x in range(6)]
        logZ, particles = SMC(n_particles, exp)

        print('logZ: ', logZ)

        values = torch.stack(particles)
        #TODO: some presentation of the results
