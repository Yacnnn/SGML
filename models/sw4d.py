import os
import ot
import numpy as np
import scipy.io as sio
import tensorflow as tf 
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from models.xw4d import Xw4d
from tqdm import tqdm 
from utils import utils

from utils.process import uniform_transport_matrix

class Sw4d(Xw4d):
    def __init__(self, 
                 gcn_type = "krylov-4",
                 store_apxf = False,
                 l2reg = 0,
                 loss_name = "NCA",
                 num_of_layer = 2,
                 hidden_layer_dim = 0,
                 final_layer_dim = 0,                
                 nonlinearity = "relu",
                 sampling_type = "regular",
                 gcn_extra_parameters = {},
                 num_of_theta_sampled = 1, 
                 dataset = ""                 
                ):
        super(Sw4d, self).__init__(
                                   gcn_type = gcn_type,
                                   l2reg = l2reg ,
                                   loss_name = loss_name,
                                   num_of_layer = num_of_layer,
                                   hidden_layer_dim = hidden_layer_dim,
                                   final_layer_dim = final_layer_dim,                
                                   nonlinearity = nonlinearity,
                                   store_apxf = store_apxf,
                                   gcn_extra_parameters = gcn_extra_parameters,
                                   sampling_type = sampling_type,
                                   num_of_theta_sampled = num_of_theta_sampled , 
                                   dataset = dataset  ) 

    def call(self, feat, adj, lab, ind):
        lab = tf.convert_to_tensor(lab)[np.newaxis,:]
        return self.loss( self.square_distance_quad_tf(feat, adj, ind, self.thetalist ,False ) , labels = lab ), 0

#Compute SW for a unique projection vector theta
    def square_distance_fromtheta_tf(self, feat, adj, ind, theta, display = False):
        output = self.gcn([feat,adj,ind])
        dmsq = []
        for i in tqdm(range(len(output)),disable= not display):
            for j in range(len(output)):
                if j > i : 
                    thetamin = tf.transpose( output[i] @ theta )
                    thetamax = tf.transpose( output[j] @ theta )
                    sthetamin = tf.transpose(tf.sort(thetamin))
                    sthetamax = tf.sort(thetamax)
                    key = str(output[i].shape[0]) + "-" + str(output[j].shape[0])
                    dmsq.append(tf.reduce_sum(self.transportmatrix[key] * tf.math.pow(sthetamin - sthetamax, 2)))
                else :
                    dmsq.append(0)
        distances_sq = tf.reshape(dmsq,[len(output),len(output)])
        distances_sq = distances_sq + tf.transpose(distances_sq)
        return  distances_sq

#Very slow implementation of SW; compute the square of the distance
    def square_distance_tf(self, feat, adj, thetalist, display = False):
        distances_sq =  np.zeros((len(feat),len(feat)))
        n = thetalist.shape[1]
        for t in range(n): 
            distances_sq += self.square_distance_fromtheta_tf(feat, adj, thetalist[:,t:t+1], display)/n
        return distances_sq

#Numpy - Quadratic implementation of SW; compute the square of the distance
    def square_distance_quad_np(self, feat, adj, ind, thetalist, display = False):
        n = thetalist.shape[1]
        output = self.gcn([feat,adj,ind])
        output = [np.array(a) for a in output]
        utils.tic()
        dmsq = []
        if self.sampling_type == 'basis':
            output_sort = [ np.sort(o,axis = 0) for o in output ]
        else:
            output_sort = [ np.sort(np.dot( o, thetalist) ,axis = 0) for o in output ]
        for i in tqdm(range(len(output)),disable= not display):
            for j in range(len(output)):
                if j > i :
                    # tm = self.transportmatrix[str(output[i].shape[0]) + "-" + str(output[j].shape[0])]
                    tm = uniform_transport_matrix(output[i].shape[0],output[j].shape[0])
                    D = ot.dist(output_sort[i],output_sort[j])
                    dmsq.append(tf.reduce_sum(tm*D))
                else :
                    dmsq.append(0)
        distance_sq = tf.reshape(dmsq,[len(output),len(output)])
        distance_sq = distance_sq + tf.transpose(distance_sq)
        utils.toc()
        return distance_sq/n

#Numpy - Quadratic implementation of SW; compute the square of the distance
    def square_distance_quad_tf(self, feat, adj, ind, thetalist, display = False):
        n = thetalist.shape[1]
        output = self.gcn([feat,adj,ind])
        dmsq = []
        if self.sampling_type == 'basis':
            output_sort = [ tf.sort(o,axis = 0) for o in output ]
        else:
            output_sort = [ tf.sort( o @ thetalist ,axis = 0) for o in output ]
        for i in tqdm(range(len(output)),disable= not display):
            for j in range(len(output)):
                if j > i :
                    # tm = self.transportmatrix[str(output[i].shape[0]) + "-" + str(output[j].shape[0])]
                    tm = uniform_transport_matrix(output[i].shape[0],output[j].shape[0])
                    L = tf.reduce_sum(output_sort[i]*output_sort[i], axis = 1)[:,np.newaxis] 
                    C = tf.reduce_sum(output_sort[j]*output_sort[j], axis = 1)[np.newaxis,:]
                    D = tf.nn.relu(L - 2*output_sort[i] @ tf.transpose(output_sort[j]) + C)
                    dmsq.append(tf.reduce_sum(tm*D))
                else :
                    dmsq.append(0)
        distance_sq = tf.reshape(dmsq,[len(output),len(output)])
        distance_sq = distance_sq + tf.transpose(distance_sq)
        return distance_sq/n 

    def distance_tf(self, feat, adj, ind = -1, display = None):
        theta = self.thetalist
        return np.sqrt(self.square_distance_tf(feat, adj, ind, theta))

    def distance_quad_np(self, feat, adj, ind = -1, display = None):
        theta = self.thetalist
        return np.sqrt(self.square_distance_quad_np(feat, adj, ind, theta))

    def distance_quad_tf(self, feat, adj, ind = -1, display = None):
        theta = self.thetalist
        return tf.sqrt(self.square_distance_quad_tf(feat, adj, ind, theta))
