#an algorithm which takes as an argument any probability density function (not necessarily normalized) and returns a collection of samples from the normalized distribution
import numpy as np
import pylab as plt
import scipy

from scipy.stats import norm
from scipy.stats import expon

gauss = lambda x: 10*norm.pdf(x)

sumgauss = lambda x: 10*norm.pdf(x) + 15*norm.pdf(x,loc = 20, scale = 4)

exp = lambda x: 10*expon.pdf(x)

exgauss = lambda x: 10*expon.pdf(x) + 15*norm.pdf(x,loc = 20, scale = 4)


def metrop(dist,full_out=False):
    #TODO look up how to initialize markov-chain?

    init = 10
    sigma = 1
    samples = []
    accepted = 0.
    logdis = lambda x: np.log(dist(x))
    count = 0

    while True:
        x_prime = np.random.normal(init,5)
        a = dist(x_prime)/dist(init)

        if logdis(x_prime)>=logdis(init):
            init = x_prime
            count+=1
        else:
            i = np.random.rand()
            if i<=a:
                init = x_prime
                count+=1
            else:
                pass


        if count>100:
            break

    samples+=[x_prime]

    for u in range(10000):
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

def test(samples,dist):
    #attempt to compute k-l divergence
    #d(P|Q) = \int(p(x)*log(p(x)/q(x))dx)

    #approximate by riemann sum
    samples = np.array(samples)
    n = 1000

    x = np.linspace(samples.min(),samples.max(),n)

    px = np.array([dist(xx) for xx in x[:n-1]])

    #approximate q(x) by (number of samples in (x,x+1))/(len(samples)
    sampmatrix = samples*np.ones((n,1))
    qx = [[x[u]<=a<x[u+1] for a in sampmatrix[u]] for u in range(len(x)-1)]
    qx = np.array(qx).sum(axis=1)/(1.*len(samples))

    nonzeroq = qx.nonzero()[0]
    qx = qx[nonzeroq]
    px = px[nonzeroq]

    l = (samples.max()-samples.min())/len(qx)

    nonzerop = px.nonzero()[0]
    qx = qx[nonzerop]
    px = px[nonzerop]
     
    evals = (l*(px*np.log(px/qx)))
    dpq = np.sum(evals)

    return dpq,evals

    
