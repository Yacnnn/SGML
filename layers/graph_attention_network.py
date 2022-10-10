import numpy as np
import tensorflow as tf
from .graph_attention_layer import GraphAttention
# from tensorflow.python.keras.layers import Dropout
from utils import process
from tqdm import tqdm 

class Graph_attention_network(tf.keras.Model):
        
    def __init__(self,
                 nb_of_layer,
                 output_features_size,
                 learning_rate,
                 dropout_rate,
                 n_attn_heads,
                 l2_reg,
                 batch_size = 2
                 ):

        super(Graph_attention_network,self).__init__()

        #Parameters
        self.dropout_rate = dropout_rate
        self.nb_of_layer = nb_of_layer
        self.learning_rate = learning_rate
        self.output_features_size  = output_features_size
        self.n_attn_heads = n_attn_heads
        self.l2_reg = l2_reg
        self.batch_size = batch_size

        #Layers
        self.gals = []
        self.drops = []

        for layer  in range(nb_of_layer):
            # self.drops.append(Dropout(dropout_rate))
            self.gals.append(GraphAttention(self.output_features_size,#[layer],
                                    attn_heads=self.n_attn_heads,#[layer],
                                    attn_heads_reduction='concat',
                                    dropout_rate=dropout_rate,
                                    activation='elu',
                                    kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg),
                                    attn_kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg)))
                                
    def call(self,inputs):
        
        layers_output = [[] for k in range(self.nb_of_layer+1)]

        for inputs_ in zip(inputs[0],inputs[1]):

            inputs_feat = inputs_[0]
            inputs_adj = inputs_[1]

            output = inputs_feat
            layers_output[0].append(output)

            for layer in range(self.nb_of_layer):
                output = self.gals[layer]([output,inputs_adj])
                layers_output[layer+1].append(output) 

        # layers_output = [ tf.concat(layers_output[k],axis=0) for k in range(self.nb_of_layer+1)]
        layers_output = layers_output[-1]
        return layers_output

    # def create_batch(self,data):
    #     graph_data = [data["node_labels"],data["graph_adjency_matrix"] ]
    #     batch, batch_indice = process.create_batch(graph_data,nb_graph = graph_data[0].shape[0],minibatch_size=self.batch_size,shuffle=True)
    #     batch_labels = [process.batch_labels(x) for x in batch[0]]
    #     batch_feat = [ [ tf.convert_to_tensor(item) for item in tab] for tab in batch[0]]
    #     batch_adj = [ [ tf.convert_to_tensor(item,dtype=tf.float64) for item in tab] for tab in batch[1]]
    #     batch_inputs = zip(tqdm(batch_feat,unit = 'batch'),batch_adj)
    #     return batch_inputs, batch_indice, batch_labels