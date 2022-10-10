import numpy as np
import ot
from utils.process import uniform_transport_matrix, uniform_pw
from utils import utils
import time 
import scipy.io as sio

def timeis(func):
    '''Decorator that reports the execution time.'''
    def wrap(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        delta_time = end - start
        print(func.__name__, delta_time)
        return result, delta_time
    return wrap

def generate_random_measures(n_samples, dimension):
    return np.random.normal(size = (n_samples, dimension))

@timeis
def wwl(distrib1, distrib2, sinkhorn_lambda = 0):
    costs = ot.dist(distrib1, distrib2, metric = 'sqeuclidean')
    if sinkhorn_lambda > 0:
        mat = ot.sinkhorn(np.ones(len(distrib1))/len(distrib1), 
                            np.ones(len(distrib2))/len(distrib2), costs, sinkhorn_lambda, numItermax =  1000)
        return np.sum(np.multiply(mat, costs))
    else:
        return ot.emd2([], [], costs) 

@timeis
def sw(distrib1, distrib2, M = 50):
    return ot.sliced.sliced_wasserstein_distance(distrib1, distrib2, n_projections=M, p=2)

@timeis
def rpswv_sq(distrib1, distrib2):
    d = distrib1.shape[1]
    argsort_distrib1 = np.argsort(distrib1,axis = 0) 
    argsort_distrib2 = np.argsort(distrib2,axis = 0) 
    D  = ot.dist(distrib1, distrib2, metric = 'sqeuclidean')
    tm = uniform_transport_matrix(len(distrib1),len(distrib2))
    Dl = [D[ argsort_distrib1[:,t] ,:][:,argsort_distrib2[:,t] ] for t in range(d)]
    return np.sum(tm*Dl)/d

@timeis
def rpswv(distrib1, distrib2):
    d = distrib1.shape[1]
    argsort_distrib1 = np.argsort(distrib1,axis = 0)
    argsort_distrib2 = np.argsort(distrib2,axis = 0)
    c = [uniform_pw(len(distrib1),len(distrib2),distrib1[argsort_distrib1[:, t],:],distrib2[argsort_distrib2[:, t],:]) for t in range(d)]
    return np.sum(c)/d

if __name__ == '__main__':
    n_samples_list = np.logspace(1, 7, num=18, endpoint=True, base=10.0, dtype=int, axis=0)
    # n_samples_list = np.logspace(1, 8, num=19, endpoint=True, base=10.0, dtype=int, axis=0)
    C = np.zeros((6,len(n_samples_list)))
    C[5,:] = n_samples_list
    for i, n_samples in enumerate(n_samples_list):
        print(n_samples)
        dimension = 5
        distrib1 = generate_random_measures(n_samples, dimension)
        distrib2 = generate_random_measures(n_samples, dimension)
        if C[0,i] == 0 and n_samples < 20000:
            C[0,i] = wwl(distrib1, distrib2)[-1]
        if C[1,i] == 0 and n_samples < 20000:
            C[1,i] = wwl(distrib1, distrib2, sinkhorn_lambda = 100)[-1]
        if C[2,i] == 0 and n_samples < 6812925 :#and n_samples < 6812925
            C[2,i] = sw(distrib1, distrib2)[-1]
        if C[3,i] == 0 and n_samples < 20000:
            C[3,i] = rpswv_sq(distrib1, distrib2)[-1]
        if C[4,i] == 0 :
            C[4,i] = rpswv(distrib1, distrib2)[-1]
        print(i,n_samples)
        sio.savemat('runtimes.mat',{
            'runtimes_matrix' : C
        })
    print(C)





