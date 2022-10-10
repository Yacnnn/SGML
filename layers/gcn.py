import numpy as np
import tensorflow as tf

class Gcn(tf.keras.Model):
        
    def __init__(self,
                num_of_layer = 2,
                nonlinearity = "relu",
                hidden_dim = 10,
                output_dim = 10,
                trainable = True,
                concat = False,
                l2reg = 0,
                 ):
        super(Gcn,self).__init__()
        #Parameters
        self.num_of_layer = num_of_layer;
        self.nonlinearity = nonlinearity
        self.hidden_dim = hidden_dim     
        self.output_dim = output_dim
        self.trainable = trainable
        self.l2reg = l2reg
        self.concat = concat
        #Layer
        if self.num_of_layer > 0: 
            layers_dim = [hidden_dim]*(self.num_of_layer-2) + [output_dim]
            self.dense_layer = [None]+[ tf.keras.layers.Dense(layers_dim[k-1], activation = self.nonlinearity_func() , trainable = True, use_bias = False) for k in range(1,self.num_of_layer)]
        elif self.num_of_layer == -1:
            self.dense_layer = [ tf.keras.layers.Dense(self.output_dim, activation = self.nonlinearity_func() , trainable = True, use_bias = False)]
        else:
            pass

    def call(self,inputs):
        if self.num_of_layer == 0 :
                return inputs[0]
        outputs = []
        for inputs_ in zip(inputs[0],inputs[1]):
            inputs_feat = inputs_[0]
            inputs_adj = inputs_[1] + tf.eye(tf.shape(inputs_[1])[0])
            D = tf.sqrt(tf.linalg.diag(tf.math.reciprocal(tf.reduce_sum(inputs_adj,axis=0))))
            inputs_adj = D @ inputs_adj @ D
            if self.num_of_layer == -1 :
                outputs.append(self.dense_layer[0]( tf.linalg.expm(inputs_adj) @ inputs_feat) )
            else : 
                output = inputs_feat
                output_tab = [output]
                for layer in range(1,self.num_of_layer):
                    output = self.dense_layer[layer](  inputs_adj @ output )
                    output_tab.append(output)
                if self.concat:
                    outputs.append(tf.concat(output_tab, axis = 1))
                else:
                    outputs.append(output_tab[-1])
        return outputs
    
    def nonlinearity_func(self):
        if self.nonlinearity == "relu":
            return tf.nn.relu
        if self.nonlinearity == "tanh":
            return tf.nn.tanh
        if self.nonlinearity == None:
            return None