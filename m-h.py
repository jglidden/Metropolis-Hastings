#an algorithm which takes as an argument any probability density function (not necessarily normalized) and returns a collection of samples from the normalized distribution
import numpy as np
import pylab as plt
import scipy

from scipy.stats import norm

gauss = lambda x: 10*norm.pdf(x)

sumgauss = lambda x: 10*norm.pdf(x) + 15*norm.pdf(x,loc = 20, scale = 4)

def metrop(dist,full_out=False):
    #TODO look up how to initialize markov-chain?

    init = 10
    sigma = 1
    samples = [init]
    accepted = 0.
    logdis = lambda x: np.log(dist(x))
    for u in range(1000):
        x_t = samples[u]
        #we will use a Gaussian centered at xt for Q
        x_prime = np.random.normal(x_t,5)

        a = dist(x_prime)/dist(x_t)
        #print a

        if logdis(x_prime)>=logdis(x_t):
            samples+=[x_prime]
            accepted+=1.
        else:
            i = np.random.rand()
            if i<=a:
                samples+=[x_prime]
                accepted+=1.
            else:
                samples+=[x_t]
    info = accepted/1000

    if full_out==True:
        return samples,info
    else:
        return samples

#TODO extend to multiple dimensions

#def test(samples,dist):
#    #attempt to compute k-l divergence
#    #d(P|Q) = \int(p(x)*log(p(x)/q(x))dx)
#    d = np.sum(np.log(dist(x)








    






