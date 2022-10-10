import numpy as np
import tensorflow as tf

class Krylov(tf.keras.Model):
    def __init__(self,
                 output_dim = 3,
                 num_of_layer = 3,
                 hidden_layer_size = 4,
                 keep_prob = 1,
                 krylov_deep = 4,
                 use_common_hidden_layer = True,
                 l2_reg = 0,
                 ):
        super(Krylov,self).__init__()
        #Parameters
        self.output_dim = output_dim
        self.num_of_layer = num_of_layer
        self.hidden_layer_size  = hidden_layer_size
        self.keep_prob = keep_prob
        self.krylov_deep = krylov_deep
        self.use_common_hidden_layer = use_common_hidden_layer
        self.l2_reg = l2_reg

    def build(self, input_shape):
        #Create layers
        self.krylovs = []
        self.layers_dropout = []
        for layer in range(self.num_of_layer):
            if layer != 0 :
                self.krylovs.append(self.add_weight(shape=(self.hidden_layer_size*self.krylov_deep,self.hidden_layer_size), trainable=True, name = "cdgn_phi_kryl"))
                self.layers_dropout.append(tf.keras.layers.Dropout(1-self.keep_prob, name = "cdgn_phi_kryl"))
            else:
                self.krylovs.append([])
                self.layers_dropout.append(tf.keras.layers.Dropout(1-self.keep_prob, name = "cdgn_phi_kryl"))
        self.krylovs[0] = self.add_weight(shape=(input_shape[0][0][1]*self.krylov_deep,self.hidden_layer_size),trainable=True,name = "cdgn_phi_kryl")
        self.mean = tf.keras.layers.Dense(self.output_dim, activation = "relu")#, kernel_initializer = "zeros")
        # self.logvar = tf.keras.layers.Dense(self.output_dim, activation = None, kernel_initializer = "zeros", name = "cdgn_phi_kryl" )
        
    def call(self,inputs):       
        layers_output = [[] for k in range(self.num_of_layer)]
        for inputs_ in zip(inputs[0],inputs[1]):
            inputs_feat = inputs_[0]
            inputs_adj = inputs_[1]
            D = tf.sqrt(tf.linalg.diag(tf.math.reciprocal(tf.reduce_sum(inputs_adj,axis=0))))
            inputs_adj = D @ inputs_adj @ D
            output = inputs_feat
            for layer in range(self.num_of_layer):
                tab = []
                tab.append(output)
                for k in range(1,self.krylov_deep):
                    tab.append(tf.matmul(inputs_adj,tab[k-1]))
                output = self.layers_dropout[layer](tf.math.tanh(tf.matmul(tf.concat(tab,axis=1),self.krylovs[layer])))
                layers_output[layer].append(output)     

        layers_output= [ self.mean(tf.concat([layers_output[k][i]  for k in range(self.num_of_layer) ] ,axis=1)) for i in range(len(layers_output[0])) ]
        # layers_output_ = [tf.concat(layers_output[k],axis=0) for k in range(self.num_of_layer)]
        # layers_output = tf.concat(layers_output_ , axis = 1)
        return layers_output