import numpy as np
import tensorflow as tf

from utils.process import memoize_compute_features
# same as SGCN with a MLP instead of only a linear map 

class Sgcn_plus(tf.keras.Model):
        
    def __init__(self,
                num_of_layer = 2,
                nonlinearity = "relu",
                output_dim = 10,
                trainable = True,
                l2reg = 0,
                store_apxf = False,
                gcn_extra_parameters = {}
                 ):
        super(Sgcn_plus,self).__init__()
        #Parameters
        self.num_of_layer = num_of_layer
        self.nonlinearity = nonlinearity
        self.output_dim = output_dim
        self.trainable = trainable
        self.l2reg = l2reg
        self.store_apxf = store_apxf
        self.gcn_e_p =  { #Defaut dictionnary
            'hlayer' : 2*[output_dim],
            'hnonlinearity' : self.str2nonlinearity('relu'),
            'fnonlinearity' : self.str2nonlinearity('relu')
        } | gcn_extra_parameters
        #Layer
        self.dense_layers = [tf.keras.layers.Dense(h, activation = self.str2nonlinearity(self.gcn_e_p['hnonlinearity']) , trainable = True, use_bias = True)
        for h in self.gcn_e_p['hlayer']] + [tf.keras.layers.Dense(self.output_dim, activation = self.str2nonlinearity(self.gcn_e_p['fnonlinearity']) , trainable = True, use_bias = True)]
        self.apply_dense_layers = lambda h, inputlayer : self.apply_dense_layers(h+1,self.dense_layers[h](inputlayer)) if h < len(self.dense_layers) else inputlayer; 
        print('')
    # @lru_cache(maxsize=None)
    @memoize_compute_features
    def compute_feats(self, input_feat, input_adj, input_ind, num_of_layer, store_apxf = False):
        input_adj = input_adj + tf.eye(tf.shape(input_adj)[0])
        D = tf.sqrt(tf.linalg.diag(tf.math.reciprocal(tf.reduce_sum(input_adj,axis=0))))
        input_adj = D @ input_adj @ D
        return np.linalg.matrix_power(input_adj, num_of_layer) @ input_feat

    def call(self,inputs): # inputs = [feat, adj, ind]
        if self.num_of_layer == 0 :
                return inputs[0]
        outputs_ = []
        for input_ in zip(inputs[0],inputs[1],inputs[2]):
            outputs_.append(self.compute_feats(input_[0], input_[1], input_[2], self.num_of_layer , self.store_apxf))
        outputs = [self.apply_dense_layers(0, out) for out in outputs_]
        return outputs
    
    def oldcall(self,inputs): # inputs = [feat, adj, ind]
        if self.num_of_layer == 0 :
                return inputs[0]
        outputs_ = []
        for input_ in zip(inputs[0],inputs[1]):
            input_feat = input_[0]
            input_adj = input_[1] + tf.eye(tf.shape(input_[1])[0])
            D = tf.sqrt(tf.linalg.diag(tf.math.reciprocal(tf.reduce_sum(input_adj,axis=0))))
            input_adj = D @ input_adj @ D
            if self.num_of_layer == -1 :
                outputs_.append(tf.linalg.expm(input_adj) @ input_feat)
            else : 
                output = input_adj
                for layer in range(1,self.num_of_layer):
                    output = output @ input_adj
                outputs_.append(output @ input_feat)
        outputs = [self.apply_dense_layers(out) for out in outputs_]
        return outputs

    def nonlinearity_func(self):
        if self.nonlinearity == "relu":
            return tf.nn.relu
        if self.nonlinearity == "tanh":
            return tf.nn.tanh
        if self.nonlinearity == None:
            return None

    def str2nonlinearity(self, name):
        if name == 'relu':
            return tf.nn.relu
        if name == 'tanh':
            return tf.nn.tanh
        return None