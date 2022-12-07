# import n_sphere
import numpy as np
from dppy.multivariate_jacobi_ope import MultivariateJacobiOPE
import ot

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

# For uniform distribution 
def uniform_part_pw(p_nbins, q_nbins, pfeatures, qfeatures):
    """ pw = sqrt( (1/M) sum_{theta}  pw_part_theta^2 )  
        Compute squared part of PW (--> pw_part_theta^2 <--) for one projection theta of uniform distribution assuming that pfeatures and qfeatures rows have been already sorted
        according to this projectiion """
    c = 0
    i = 0
    j = 0
    w_i = 1/p_nbins
    w_j = 1/q_nbins
    while True:
        if w_i < w_j :
            # utm[i,j] = w_i
            c = c + w_i*np.linalg.norm(pfeatures[i,:] - qfeatures[j,:])**2
            i += 1
            if i == p_nbins:
                break
            w_j -= w_i
            w_i = 1/p_nbins
        else  :
            # utm[i,j] = w_j
            c = c + w_j*np.linalg.norm(pfeatures[i,:] - qfeatures[j,:])**2
            j +=1 
            if j == q_nbins:
                break
            w_i -= w_j
            w_j = 1/q_nbins
    return c #utm

# For distribution possibly non uniform
def part_pw(p_weights, q_weights, pfeatures, qfeatures):
    """ pw = sqrt( (1/M) sum_{theta}  pw_part_theta^2 ) 
        Compute squared part of PW (--> pw_part_theta^2 <--) for one projection theta of distribution assuming that pfeatures and qfeatures rows have been already sorted
        according to this projectiion
        p_weights[i] is the weights associated to features pfeatures[i,:]
        q_weights[j] is the weights associated to features pfeatures[j,:]

    """
    c = 0
    i = 0
    j = 0
    p_nbins = len(p_weights)
    q_nbins = len(q_weights)
    w_i = p_weights[i]
    w_j = p_weights[j]
    while True:
        if w_i < w_j :
            c = c + w_i*np.linalg.norm(pfeatures[i,:] - qfeatures[j,:])**2
            i += 1
            if i == p_nbins:
                break
            w_j -= w_i
            w_i = p_weights[i]
        else  :
            c = c + w_j*np.linalg.norm(pfeatures[i,:] - qfeatures[j,:])**2
            j +=1 
            if j == q_nbins:
                break
            w_i -= w_j
            w_j = q_weights[i]
    return c #utm

def compute_pw(distrib1, distrib2, weights1, weights2, restricted = True, sampling_number = 50):
    """ Implementation of (R)PW with for distribution non necessary uniform
        It is never used on this repository since we deal only with uniform distribution 
        But this implementation is provided if needed 
        distrib1 : p x n (features size x number of bins)
        distrib2 : p x n'
        weights1, weights2 : weights[i] correspond to weigth of bin distrib[:, i] wieghts; sum(weghts) = 1
        restricted : if True compute RPW else PW
        sampling_number : Numbner of theta to sample if restricted is False
    """
    p = distrib1.shape[1]
    if restricted:
        argsort_distrib1 = np.argsort(distrib1,axis = 0)
        argsort_distrib2 = np.argsort(distrib2,axis = 0)
    else:
        T = np.random.normal(size= (p, sampling_number))
        thetalist = T / np.linalg.norm(T, axis=0)
        argsort_distrib1 = np.argsort(np.dot( distrib1, thetalist),axis = 0)
        argsort_distrib2 = np.argsort(np.dot( distrib1, thetalist),axis = 0)
    c = [part_pw(weights1,weights2,distrib1[argsort_distrib1[:, t],:],distrib2[argsort_distrib2[:, t],:]) for t in range(p)]
    return np.sqrt(np.sum(c)/p)

def uniform_transport_matrixv2(p_nbins, q_nbins, order_p, order_q):
    """ Return the transport matrix between two (sorted) real uniform distribution with p and q bins.
    """
    mn = p_nbins
    mx = q_nbins
    tampon = np.zeros((mn,mx*mn)).astype(np.float32)
    tampon2 = np.zeros((mn,mx)).astype(np.float32)
    for q in range(mn):
        tampon[q,(mx)*q:(mx)*q+mx] = 1/mx
    for p in range(mx):
        tampon2[:,p] = np.sum(tampon[:,(mn)*p:(mn)*p+mn],1)
    return tampon2

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







         











