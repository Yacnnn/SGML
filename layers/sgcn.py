import numpy as np
import tensorflow as tf

class Sgc(tf.keras.Model):
        
    def __init__(self,
                nb_of_layer,
                nonlinearity = "relu",
                output_dim = 10,
                trainable = True,
                l2reg = 0,
                 ):
        super(Sgc,self).__init__()
        #Parameters
        self.nb_of_layer = nb_of_layer
        self.nonlinearity = nonlinearity
        self.output_dim = output_dim
        self.trainable = trainable
        self.l2_reg = l2_reg
        #Layers
        self.dense_layer = tf.keras.layers.Dense(self.output_dim, activation = nonlinearity_func() , trainable = True, use_bias = False)

    def call(self,inputs):
        if self.nb_of_layer == 0 :
                return inputs[0]
        outputs_ = []
        for inputs_ in zip(inputs[0],inputs[1]):
            inputs_feat = inputs_[0]
            inputs_adj = inputs_[1] + tf.eye(tf.shape(inputs_[1])[0])
            D = tf.sqrt(tf.linalg.diag(tf.math.reciprocal(tf.reduce_sum(inputs_adj,axis=0))))
            inputs_adj = D @ inputs_adj @ D
            if self.nb_of_layer == -1 :
                outputs_.append(tf.linalg.expm(inputs_adj) @ inputs_feat)
            else : 
                output = inputs_adj
                for layer in range(1,self.nb_of_layer):
                    output = output @ inputs_adj
                outputs_.append(output @ inputs_feat)
        outputs = [self.dense_layer(out) for out in outputs_]
        return outputs
    
    def nonlinearity_func(self):
        if self.nonlinearity == "relu":
            return tf.nn.relu
        if self.nonlinearity == "tanh":
            return tf.nn.tanh
        if self.nonlinearity == None:
            return None