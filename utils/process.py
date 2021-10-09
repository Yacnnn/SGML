# import n_sphere
import numpy as np
from dppy.multivariate_jacobi_ope import MultivariateJacobiOPE

def uniform_transport_matrix(p_nbins, q_nbins):
    """ Return the transport matrix between two (sorted) real uniform distribution with p and q bins.
    """
    utm = np.zeros((p_nbins,q_nbins)).astype(np.float32)
    i = 0
    j = 0
    w_i = 1/p_nbins
    w_j = 1/q_nbins
    while True:
        if w_i < w_j :
            utm[i,j] = w_i
            i += 1
            if i == p_nbins:
                break
            w_j -= w_i
            w_i = 1/p_nbins
        else  :
            utm[i,j] = w_j
            j +=1 
            if j == q_nbins:
                break
            w_i -= w_j
            w_j = 1/q_nbins
    return utm

def list_of_primes(n):
    """ Return the first n prime number.
    """
    primes = [2]
    y = 2
    while len(primes) < n:
        y = y + 1
        for z in range(2, y):
            if y % z == 0:
                break
        else:
            primes.append(y)
    return primes

def hammersley_compoments(n , base):
    """ Return the component related to base b of Hammersley sequence of n points.  
        n is the number of points to generate 
        base (0 or a prime number) is the base
        If base == 0, this function return the first component (or the last depending on the point of view) of the hammersley sequence : [1,...,n]/n 
    """
    if base == 0 : 
        seq = np.arange(1,n+1)/n
    else : 
        seq = np.zeros((1,n))
        seed = np.arange(1,n+1)
        base_inv = 1.0 / base
        while np.any( seed != 0) :
            digit = seed % base
            seq = seq + digit * base_inv
            base_inv = base_inv / base
            seed = np.floor(seed / base)
    return seq

def hammersley_sequence(n, p):
    """ Return n vectors (dimension p) of the Hammersley sequence related to the p - 1 first prime number. 
    """
    sequence = np.zeros((p, n))
    primes = [0] + list_of_primes(p - 1)
    for k in range(p):
        sequence[k,:] = hammersley_compoments(n , primes[k])
    return sequence

def hypcube2hypsphere(sequence):
    rescale_seq = 2*sequence-1
    vecs = [] 
    for k in range(rescale_seq.shape[1]):
        angles = np.arctan2(  np.sqrt( np.cumsum( (rescale_seq[:,k]**2)[:0:-1])[::-1]), rescale_seq[:,k][:-1] )
        temp = np.array([1] + list(np.cumprod(np.sin(angles))))
        temp[:-1] = np.cos(angles)*temp[:-1]
        vec = temp
        vecs.append(vec[:,np.newaxis])
    vecs = np.concatenate(vecs, axis = 1)
    return vecs

def hammersley_sphere_seq(n, p):
    sequence = hammersley_sequence(n, p)
    return hypcube2hypsphere(sequence)

def dpp_sphere_smpl(n, p):
    # jac_params = np.array([ [-0.5,0.5] for k in range(p)])
    jac_params =  -0.5 + np.random.rand(p, 2)
    dpp = MultivariateJacobiOPE(n, jac_params)
    samples = (np.transpose(dpp.sample()) + 1.0) / 2.0
    return hypcube2hypsphere(samples)

    
    





         











