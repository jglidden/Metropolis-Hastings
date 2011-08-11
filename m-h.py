#an algorithm which takes as an argument any probability density function (not necessarily normalized) and returns a collection of samples from the normalized distribution
import numpy as np
#import pylab as plt

def pdis(x):
    #this will be our unnormalized function (for now)
    if 0<x<1000:
        y = 1000-x
    else:
        y = 0

    return y

def metrop(func = pdis):
    #TODO look up how to initialize markov-chain?

    init = 1
    samples = [init]
    for u in range(1000):
        #TODO more intelligent way to figure out how long to run
        x_t = samples[u]
        #we will use a Gaussian centered at xt for Q
        #TODO pick sigma intelligently
        sigma = 5
        x_prime = np.random.normal(x_t,5)

        a = func(x_prime)/func(x_t)

        if a>=1:
            samples+=[x_prime]
        else:
            i = np.random.rand()
            if i<=a:
                samples+=[x_prime]
            else:
                samples+=[x_t]

    return samples

#TODO extend to multiple dimensions


    






