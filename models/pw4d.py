import os
import ot
import numpy as np
import scipy.io as sio
import tensorflow as tf 
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from models.xw4d import Xw4d
from tqdm import tqdm 
# from utils.process import uniform_transport_matrix

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
        return self.loss( self.square_distance_fast(feat, adj, self.thetalist) , labels = lab), 0
        # return self.loss( self.square_distance(feat, adj, self.thetalist) , labels = lab), 0

    
    def square_distance_fromtheta(self, feat, adj, theta, display = False):
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
                    # keyrev = str(output[j].shape[0]) + "-" + str(output[i].shape[0])
                    # if key not in self.transportmatrix.keys():
                    #     self.transportmatrix[key] = uniform_transport_matrix(output[i].shape[0],output[j].shape[0])
                    #     self.transportmatrix[keyrev] = tf.transpose(self.transportmatrix[key])
                    L = tf.reduce_sum(sthetamin*sthetamin, axis = 1)[:,np.newaxis] #tf.linalg.tensor_diag_part(sthetamin @ tf.transpose(sthetamin))[:,np.newaxis]
                    C = tf.reduce_sum(sthetamax*sthetamax, axis = 1)[np.newaxis,:] # tf.linalg.tensor_diag_part(sthetamax @ tf.transpose(sthetamax))[np.newaxis,:]
                    distance_temp = tf.nn.relu(L - 2*sthetamin @ tf.transpose(sthetamax) + C)
                    dmsq.append(tf.reduce_sum(self.transportmatrix[key] * (distance_temp) ))
                    # transportmatrix = uniform_transport_matrix(output[i].shape[0],output[j].shape[0])
                    # dmsq.append(tf.reduce_sum(transportmatrix * (distance_temp) ))

                else :
                    dmsq.append(0)
        distance_sq = tf.reshape(dmsq,[len(output),len(output)])
        distance_sq = distance_sq + tf.transpose(distance_sq)
        return distance_sq

    def square_distance(self, feat, adj, thetalist, display = False):
        distance_sq =  np.zeros((len(feat),len(feat)))
        n = thetalist.shape[1]
        for t in range(n): 
            distance_sq += self.square_distance_fromtheta(feat, adj, thetalist[:,t:t+1], display)/n
        return distance_sq

    def square_distance_fast(self, feat, adj, thetalist, display = False):
        ng = len(feat)
        D = tf.zeros((ng,ng))
        n = thetalist.shape[1]
        if adj == [] :
            output = feat
        else:
            output = self.gcn([feat,adj])
        nf = output[0].shape[1]
        
        flen = [len(f) for f in output]
        themax = max(flen)
        up_feat = [f @ self.thetalist for f in output]
        # up_sfeat = [tf.sort(f, axis = 0) for f in up_feat  ]
        # H = tf.convert_to_tensor([tf.concat( [f, tf.zeros([themax-f.shape[0],f.shape[1]])] , axis = 0) for f in up_sfeat])
       
        up_asfeat = [tf.argsort(f, axis = 0) for f in up_feat  ]
        argH = tf.convert_to_tensor([tf.concat( [f,   f.shape[0] - 1 + tf.cumsum(tf.ones((themax-f.shape[0],f.shape[1]), dtype=tf.dtypes.int32 ),axis = 0)     ] , axis = 0) for f in up_asfeat])
        up_sfeat = tf.convert_to_tensor([ tf.concat( [o, tf.zeros((themax - o.shape[0],o.shape[1]))] , axis = 0) for o in  output ])
       
        H = tf.transpose(tf.gather(up_sfeat, argH, batch_dims = 1), [  0 , 1, 3,2])
        rH = tf.reshape(H, [-1, H.shape[-2]* H.shape[-1]])   

        # rH2 = tf.reshape(H2, [-1, H.shape[-1]])   
        rH0 = rH[:,np.newaxis,:]
        rH1 = rH[np.newaxis,:,:]
        DrH_pow = tf.math.pow(rH0-rH1, 2)
        DrH = tf.convert_to_tensor([tf.split(S, ng , axis = 1) for S in tf.split(DrH_pow, ng)])
        DrH = tf.reshape(DrH, [ng, ng, themax, themax, nf, -1])
        DrH = tf.reduce_sum(DrH, axis = 4)
        # DrH = np.sum(DrH, axis = 4)
        # DrH2 = tf.convert_to_tensor([tf.split(S, ng*nf , axis = 1) for S in tf.split(DrH2_pow, ng*nf)])

        # rH = tf.reshape(H, [-1, H.shape[-1]])    
        # rH0 = rH[:,np.newaxis,:]
        # rH1 = rH[np.newaxis,:,:]
        # DrH_pow = tf.math.pow(rH0-rH1, 2)
        # DrH = tf.convert_to_tensor([tf.split(S, ng , axis = 1) for S in tf.split(DrH_pow, ng)])

        integers = tf.cumsum(tf.ones((1,len(output)*(len(output)+1)//2)),axis=1)
        integers_triu = tf.squeeze(tfp.math.fill_triangular(integers, True))
        up_tri_indicesx,up_tri_indicesy = np.triu_indices(len(output))
        reorder_integers = [ int(integers_triu[ix,iy]) - 1 for ix, iy in zip(up_tri_indicesx,up_tri_indicesy)]
        arg_reorder_integers = np.argsort(reorder_integers)
        indices_reg = [[a,b]  for a, b in zip(up_tri_indicesx,up_tri_indicesy) ]
        indices = [ indices_reg[r]  for r in arg_reorder_integers]

        DrH_flat = tf.gather_nd(DrH,indices)
        tm = self.build_tm(output, themax)[:,:,:,np.newaxis]
        tm = tf.gather(tm,arg_reorder_integers)
        Dtemp = tf.reduce_sum(tf.math.multiply(tm,DrH_flat),axis = [-3, -2,-1])
        D = tf.squeeze(tfp.math.fill_triangular(Dtemp, True))
        D = D + tf.transpose(D)
        return D/n

    def square_distance_fastv2(self, feat,feat2, adj, adj2, thetalist, display = False):
        ng = len(feat)
        nd = len(feat2)
        D = tf.zeros((ng,nd))
        n = thetalist.shape[1]
        if adj == [] or adj2 == []:
            output = feat
            output2 = feat2
        else:
            output = self.gcn([feat,adj])
            output2 = self.gcn([feat2,adj2])
        nf = output[0].shape[1]       
        flen = [len(f) for f in output ]+[len(f) for f in output2]
        themax = max(flen)

        up_feat = [f @ self.thetalist for f in output]       
        up_asfeat = [tf.argsort(f, axis = 0) for f in up_feat  ]
        argH = tf.convert_to_tensor([tf.concat( [f,   f.shape[0] - 1 + tf.cumsum(tf.ones((themax-f.shape[0],f.shape[1]), dtype=tf.dtypes.int32 ),axis = 0)     ] , axis = 0) for f in up_asfeat])
        up_sfeat = tf.convert_to_tensor([ tf.concat( [o, tf.zeros((themax - o.shape[0],o.shape[1]))] , axis = 0) for o in  output ])

        up_feat2 = [f @ self.thetalist for f in output2]       
        up_asfeat2 = [tf.argsort(f, axis = 0) for f in up_feat2  ]
        argH2 = tf.convert_to_tensor([tf.concat( [f,   f.shape[0] - 1 + tf.cumsum(tf.ones((themax-f.shape[0],f.shape[1]), dtype=tf.dtypes.int32 ),axis = 0)     ] , axis = 0) for f in up_asfeat2])
        up_sfeat2 = tf.convert_to_tensor([ tf.concat( [o, tf.zeros((themax - o.shape[0],o.shape[1]))] , axis = 0) for o in  output2 ])
       
        H = tf.transpose(tf.gather(up_sfeat, argH, batch_dims = 1), [  0 , 1, 3,2])
        rH = tf.reshape(H, [-1, H.shape[-2]* H.shape[-1]])   
        rH = rH[:,np.newaxis,:]

        H2 = tf.transpose(tf.gather(up_sfeat2, argH2, batch_dims = 1), [  0 , 1, 3,2])
        rH2 = tf.reshape(H2, [-1, H2.shape[-2]* H2.shape[-1]])   
        rH2 = rH2[np.newaxis,:,:]

        DrH_pow = tf.math.pow(rH-rH2, 2)
        try :
            DrH = tf.convert_to_tensor([tf.split(S, nd , axis = 1) for S in tf.split(DrH_pow, ng)])
        except :
            print("e")
        DrH = tf.reshape(DrH, [ng, nd, themax, themax, nf, -1])
        DrH = np.sum(DrH, axis = 4)

        tm = self.build_tmv2(output , output2, themax)[:,:,:,np.newaxis]
        tm = tf.transpose(tf.reshape(tm, [ng,nd,themax,themax]), [0, 1, 2, 3])[:,:,:,:,np.newaxis]
        D = tf.reduce_sum(tf.math.multiply(tm,DrH),axis = [-3, -2,-1])
        return D/n
        
    def distance(self, feat, adj, theta, display):
        return tf.sqrt(square_distance(feat, adj, theta, distance))

    def distance_fast(self, feat, adj, thetalist, display):
        return tf.sqrt(self.square_distance_fast(feat, adj, thetalist, display))

    def distance_fastv2(self, feat, adj):
        feat = feat#[:400]
        adj = adj#[:400]
        output = self.gcn([feat,adj])

        ind = np.argsort([len(o) for o in output]) 
        output = [ output[i] for i in ind]

        ng = len(feat)
        D = np.zeros((ng,ng))
        DD = []
        # coeff = len(feat)//40
        # i = len(feat)//coeff
        i = 10
        # i = len(feat)//400
        limit = list(np.arange(0,ng,i))+[ng] 
        limit_down = limit[:-1]
        limit_up = limit[1:]
        if limit_up[-1] % i ==1 :
            del limit_down[-1]
            limit_up[-2] = limit_up[-1]
            del limit_up[-1]
        for hd, hu in zip(limit_down, tqdm(limit_up)):
            for vd, vu in zip(limit_down,limit_up):
                if hd == vd :
                    d = tf.sqrt(self.square_distance_fast(output[hd:hu], [], self.thetalist, display = True))
                    D[hd:hu,vd:vu] = d/2
                if hd < vd :
                    d = tf.sqrt(self.square_distance_fastv2(output[hd:hu], output[vd:vu], [], [], self.thetalist, display = True))
                    D[hd:hu,vd:vu] = d
        D = D + np.transpose(D)
        D = D[:,np.argsort(ind)]
        D = D[np.argsort(ind),:]
        return tf.convert_to_tensor(D)
