import os
import ot
import numpy as np
import scipy.io as sio
import tensorflow as tf 
import matplotlib.pyplot as plt
from models.xw4d import Xw4d
from tqdm import tqdm 

class Sw4d(Xw4d):
    def __init__(self, 
                 gcn_type = "krylov-4",
                 l2reg = 0,
                 loss_name = "NCA",
                 num_of_layer = 2,
                 hidden_layer_dim = 0,
                 final_layer_dim = 0,                
                 nonlinearity = "relu",
                 sample_type = "regular",
                 num_of_theta_sampled = 1, 
                 dataset = ""                 
                ):
        super(Sw4d, self).__init__() 

    def call(self, feat, adj, lab, s):
        lab = tf.convert_to_tensor(lab)[np.newaxis,:]
        return self.loss( self.square_distance_fromtheta( feat, adj, self.theta ) , labels = lab ), 0
    
    def square_distance_fromtheta(self, feat, adj, theta, display = False):
        output = self.gcn([feat,adj])
        dmsq = []
        for i in tqdm(range(len(output)),disable= not display):
            for j in range(len(output)):
                if j > i : 
                    thetamin = tf.transpose( output[i] @ theta )
                    thetamax = tf.transpose( output[j] @ theta )
                    sthetamin = tf.transpose(tf.sort(thetamin))
                    sthetamax = tf.sort(thetamax)
                    key = str(output[i].shape[0]) + "-" + str(output[j].shape[0])
                    # keyrev = str(output[j].shape[0]) + "-" + str(output[i].shape[0])
                    # if key not in self.transportmatrix.keys():
                    #     self.transportmatrix[key] = uniform_transport_matrix(output[i].shape[0],output[j].shape[0])
                    #     self.transportmatrix[keyrev] = tf.transpose(self.transportmatrix[key])
                    dmsq.append(tf.reduce_sum(self.transportmatrix[key] * tf.math.pow(sthetamin - sthetamax, 2)))
                else :
                    dmsq.append(0)
        distances_sq = tf.reshape(dmsq,[len(output),len(output)])
        distances_sq = distances_sq + tf.transpose(distances_sq)
        return  distances_sq

    def square_distance(self, feat, adj, thetalist, display = False):
        distances_sq =  np.zeros((len(feat),len(feat)))
        n = thetalist.shape[1]
        for t in range(n): 
            distances_sq += self.square_distance_fromtheta(feat, adj, thetalist[:,t:t+1], display)/n
        return distances_sq

    def square_distance_fast(self, feat, adj, thetalist, display = False):
        ng = len(feat)
        D = tf.zeros((ng,ng))
        n = thetalist.shape[1]
        if adj == [] :
            output = feat
        else:
            output = self.gcn([feat,adj])
        flen = [len(f) for f in output]
        themax = max(flen)
        up_feat = [f @ self.thetalist for f in output]
        up_sfeat = [tf.sort(f, axis = 0) for f in up_feat  ]

        H = tf.convert_to_tensor([tf.concat( [f, tf.zeros([themax-f.shape[0],f.shape[1]])] , axis = 0) for f in up_sfeat])
        rH = tf.reshape(H, [-1, H.shape[-1]])    
        rH0 = rH[:,np.newaxis,:]
        rH1 = rH[np.newaxis,:,:]
        DrH_pow = tf.math.pow(rH0-rH1, 2)
        DrH = tf.convert_to_tensor([tf.split(S, ng , axis = 1) for S in tf.split(DrH_pow, ng)])

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
        flen = [len(f) for f in output ]+[len(f) for f in output2]
        themax = max(flen)

        up_feat = [f @ self.thetalist for f in output]
        up_sfeat = [tf.sort(f, axis = 0) for f in up_feat  ]

        up_feat2 = [f @ self.thetalist for f in output2]      
        up_sfeat2 = [tf.sort(f, axis = 0) for f in up_feat2  ]

        H = tf.convert_to_tensor([tf.concat( [f, tf.zeros([themax-f.shape[0],f.shape[1]])] , axis = 0) for f in up_sfeat])
        rH = tf.reshape(H, [-1, H.shape[-1]])    
        rH0 = rH[:,np.newaxis,:]

        H2 = tf.convert_to_tensor([tf.concat( [f, tf.zeros([themax-f.shape[0],f.shape[1]])] , axis = 0) for f in up_sfeat2])
        rH2 = tf.reshape(H2, [-1, H2.shape[-1]])    
        rH2 = rH2[np.newaxis,:,:]

        DrH_pow = tf.math.pow(rH0-rH2, 2)
        DrH = tf.convert_to_tensor([tf.split(S, ng , axis = 1) for S in tf.split(DrH_pow, ng)])

        tm = self.build_tmv2(output , output2, themax)[:,:,:,np.newaxis]
        tm = tf.transpose(tf.reshape(tm, [ng,nd,themax,themax]), [0, 1, 2, 3])[:,:,:,:,np.newaxis]
        # tm = tf.transpose(tf.reshape(tm, [ng,nd,themax,themax]), [1, 0, 2, 3, 4])
        D = tf.reduce_sum(tf.math.multiply(tm,DrH),axis = [-3, -2,-1])

        return D/n

    def distance(self, feat, adj, thetalist, display):
        return tf.sqrt(self.square_distance(feat, adj, thetalist, display))
    
    def distance_fast(self, feat, adj, thetalist, display):
        return tf.sqrt(self.square_distance_fast(feat, adj, thetalist, display))

    def distance_fastv2(self, feat, adj):
        feat = feat
        adj = adj
        output = self.gcn([feat,adj])

        ind = np.argsort([len(o) for o in output]) 
        output = [ output[i] for i in ind]

        ng = len(feat)
        D = np.zeros((ng,ng))
        DD = []
        coeff = len(feat)//40
        i = len(feat)//coeff
        i = 20
        # i = len(feat)//100
        limit = list(np.arange(0,ng,i))+[ng] 
        limit_down = limit[:-1]
        limit_up = limit[1:]
        # if limit_up[-1] % 2 ==1 :
        #     limit_down[-1] = limit_down[-1] - 1
        if  not limit_up[-1] + 1 % i ==1 :
            limit_down[-1] = limit_up[-1] - i
        for hd, hu in zip(limit_down, tqdm(limit_up)):
            for vd, vu in zip(limit_down,limit_up):
                if hd == vd :
                    d = tf.sqrt(self.square_distance_fromthetalistv2(output[hd:hu], [], self.thetalist, display = True))
                    # d = tf.sqrt(self.square_distance_fromthetalistv2(feat[hd:hu], adj[hd:hu], self.thetalist, display = True))
                    # print(np.sum(d-d0))
                    D[hd:hu,vd:vu] = d/2
                    # d = tf.sqrt(self.square_distance_fromthetalistv3(feat[hd:hu], feat[vd:vu], adj[hd:hu], adj[vd:vu], self.thetalist, display = True))
                    # D[hd:hu,vd:vu] = d/2
                if hd < vd :
                    d = tf.sqrt(self.square_distance_fromthetalistv3(output[hd:hu], output[vd:vu], [], [], self.thetalist, display = True))
                    # d = tf.sqrt(self.square_distance_fromthetalistv3(feat[hd:hu], feat[vd:vu], adj[hd:hu], adj[vd:vu], self.thetalist, display = True))
                    # print(np.sum(d-d0))
                    # d = tf.linalg.set_diag(d, tf.linalg.diag_part(d)/2) 
                    D[hd:hu,vd:vu] = d
        D = D + np.transpose(D)
        D = D[:,np.argsort(ind)]
        D = D[np.argsort(ind),:]
        # D = tf.gather(D, np.argsort(ind))
        # D = tf.transpose(D)
        # D = D[:,list(np.argsort(ind))]
        # D = D[np.argsort(ind),:]
        return tf.convert_to_tensor(D)