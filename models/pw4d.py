from ctypes import util
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
from utils.process import uniform_part_pw, uniform_transport_matrix

class Pw4d(Xw4d):
    def __init__(self, 
                 gcn_type = "krylov-4",
                 l2reg = 0,
                 loss_name = "NCA",
                 num_of_layer = 2,
                 hidden_layer_dim = 0,
                 final_layer_dim = 0,                
                 nonlinearity = "relu",
                 sampling_type = "regular",
                 num_of_theta_sampled = 1, 
                 dataset = ""    
                ):
        super(Pw4d, self).__init__(
                                   gcn_type = gcn_type,
                                   l2reg = l2reg ,
                                   loss_name = loss_name,
                                   num_of_layer = num_of_layer,
                                   hidden_layer_dim = hidden_layer_dim,
                                   final_layer_dim = final_layer_dim,                
                                   nonlinearity = nonlinearity,
                                   sampling_type = sampling_type,
                                   num_of_theta_sampled = num_of_theta_sampled , 
                                   dataset = dataset  ) 

    def call(self, feat, adj, lab, s):
        lab = tf.convert_to_tensor(lab)[np.newaxis,:]
        return self.loss( self.square_distance_quad_tf(feat, adj, self.thetalist) , labels = lab), 0

#Compute PW4D for a unique projection vector theta
    def square_distance_fromtheta_tf(self, feat, adj, theta, display = False):
        output = self.gcn([feat,adj])
        dmsq = []
        for i in tqdm(range(len(output)),disable= not display):
            for j in range(len(output)):
                if j > i : 
                    thetamin = tf.transpose( output[i] @ theta )
                    thetamax = tf.transpose( output[j] @ theta )
                    argthetamin = tf.squeeze(tf.argsort( thetamin ))
                    argthetamax = tf.squeeze(tf.argsort( thetamax ))
                    sthetamin = tf.gather(output[i], argthetamin)
                    sthetamax = tf.gather(output[j], argthetamax)
                    key = str(output[i].shape[0]) + "-" + str(output[j].shape[0])
                    L = tf.reduce_sum(sthetamin*sthetamin, axis = 1)[:,np.newaxis] 
                    C = tf.reduce_sum(sthetamax*sthetamax, axis = 1)[np.newaxis,:]
                    distance_temp = tf.nn.relu(L - 2*sthetamin @ tf.transpose(sthetamax) + C)
                    dmsq.append(tf.reduce_sum(self.transportmatrix[key] * (distance_temp) ))
                else :
                    dmsq.append(0)
        distance_sq = tf.reshape(dmsq,[len(output),len(output)])
        distance_sq = distance_sq + tf.transpose(distance_sq)
        return distance_sq

#Very slow implementation of (R)PW; compute the square of the distance
    def square_distance_tf(self, feat, adj, thetalist, display = False):
        distance_sq =  np.zeros((len(feat),len(feat)))
        n = thetalist.shape[1]
        for t in range(n): 
            distance_sq += self.square_distance_fromtheta_tf(feat, adj, thetalist[:,t:t+1], display)/n
        return distance_sq

#Numpy - Quadratic implementation of (R)PW; compute the square of the distance
    def square_distance_quad_np(self, feat, adj, thetalist, display = False):
        n = thetalist.shape[1]
        output = self.gcn([feat,adj])
        output = [np.array(a) for a in output]
        utils.tic()
        dmsq = []
        if self.sampling_type == 'basis':
            output_argsort = [ np.argsort(o,axis = 0) for o in output ]
        else:
            output_argsort = [ np.argsort(np.dot( o, thetalist) ,axis = 0) for o in output ]
        for i in tqdm(range(len(output)),disable= display):
            for j in range(len(output)):
                if j > i :
                    # tm = self.transportmatrix[str(output[i].shape[0]) + "-" + str(output[j].shape[0])]
                    tm = uniform_transport_matrix(output[i].shape[0],output[j].shape[0])
                    D = ot.dist(output[i],output[j])
                    Dl = [D[ output_argsort[i][:,t] ,:][:,output_argsort[j][:,t] ] for t in range(n)]
                    dmsq.append(np.sum(tm*Dl))
                else :
                    dmsq.append(0)
        distance_sq = np.reshape(dmsq,[len(output),len(output)])
        distance_sq = distance_sq + np.transpose(distance_sq)
        return distance_sq/n

#Numpy - Sequential implementation of (R)PW; compute the square of the distance
    def square_distance_seq_np(self, feat, adj, thetalist, display = False):
        n = thetalist.shape[1]
        output = self.gcn([feat,adj])
        output = [np.array(a) for a in output]
        dmsq = []
        if self.sampling_type == 'basis':
            output_argsort = [ np.argsort(o,axis = 0) for o in output ]
        else:
            output_argsort = [ np.argsort(np.dot( o, thetalist) ,axis = 0) for o in output ]
        for i in tqdm(range(len(output))):
            for j in range(len(output)):
                if j > i :
                    X = [uniform_part_pw(output[i].shape[0],output[j].shape[0],output[i][ output_argsort[i][:,t] ,:],output[j][ output_argsort[j][:,t] ,:]) for t in range(n)]
                    dmsq.append(np.sum(X))
                else :
                    dmsq.append(0)
        distance_sq = np.reshape(dmsq,[len(output),len(output)])
        distance_sq = distance_sq + np.transpose(distance_sq)
        return distance_sq/n

#USE4TRAINING : Tf - Quadratic implementation of (R)PW; compute the square of the distance
    def square_distance_quad_tf(self, feat, adj, thetalist, display = False):
        n = thetalist.shape[1]
        output = self.gcn([feat,adj])
        dmsq = []
        if self.sampling_type == 'basis':
            output_argsort2 = [ np.argsort(np.argsort(o, axis = 0), axis = 0) for o in output ]
        else:
            output_argsort2 = [  np.argsort(np.argsort(np.dot( o, thetalist), axis = 0) ,axis = 0) for o in output ]
        for i in tqdm(range(len(output)),disable= not display):
            for j in range(len(output)):
                if j > i :
                    tm = uniform_transport_matrix(output[i].shape[0],output[j].shape[0])
                    # tm = self.transportmatrix[str(output[i].shape[0]) + "-" + str(output[j].shape[0])]
                    tml = [tm[ output_argsort2[i][:,t] ,:][:,output_argsort2[j][:,t] ] for t in range(n)]
                    L = tf.reduce_sum(output[i]*output[i], axis = 1)[:,np.newaxis] 
                    C = tf.reduce_sum(output[j]*output[j], axis = 1)[np.newaxis,:]
                    D = tf.nn.relu(L - 2*output[i] @ tf.transpose(output[j]) + C)
                    dmsq.append(tf.reduce_sum(tml*D))
                else :
                    dmsq.append(0)
        distance_sq = tf.reshape(dmsq,[len(output),len(output)])
        distance_sq = distance_sq + tf.transpose(distance_sq)
        return distance_sq/n 

    def distance(self, feat, adj, theta, display):
        return tf.sqrt(self.square_distance(feat, adj, theta))

    def distance_quad_np(self, feat, adj, display = None):
        theta = self.thetalist
        return np.sqrt(self.square_distance_quad_np(feat, adj, theta))

    def distance_seq_np(self, feat, adj, display = None):
        theta = self.thetalist
        return np.sqrt(self.square_distance_quad_npv2(feat, adj, theta))

    def distance_quad_tf(self, feat, adj, display = None):
        theta = self.thetalist
        return tf.sqrt(self.square_distance_quad_tf(feat, adj, theta))