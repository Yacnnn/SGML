import os 
import argparse
import ot
import tensorflow as tf
import numpy as np
import scipy.io as sio

from datetime import datetime
from tqdm import tqdm

from utils import process
from utils import process_data

from sklearn.model_selection import ParameterGrid

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
NOW = datetime.utcnow().strftime('%B_%d_%Y_%Hh%Mm%Ss')
MAX_RESTART = 1

def str2bool(string):
    """ Convert string to corresponding boolean. """
    if string in ["True","true","1"]:
        return True
    elif string in ["False","false","0"]:
        return False
    else :
        return False
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='sw4d', help='Task to execute. Only %s are currently available.'%str(process_data.available_tasks()))
    parser.add_argument('--dataset', default='MUTAG', help='Dataset to work with. Only %s are currently available.'%str(process_data.available_datasets()))
    parser.add_argument('--feature', default = 'degree', help='Features to use for nodes. Only %s are currently available.'%str(process_data.available_tasks()))
    parser.add_argument('--loss', default = "NCMML", help='Metric learning loss')
    ###    
    parser.add_argument('--gcn', type = str, default= "sgcn" , help='Type of GCN. [SGCN].')
    parser.add_argument('--num_of_layer', type = int, default= 2, help='Number of layer for GCN. [-1, integers > 0]. -1 : exponentiate. 0 : GCN = identity.')
    parser.add_argument('--hidden_layer', type = int, default = 0, help='Size of hidden layer of the GCN if applicable. [integer > 0]. O : output dimension = input dimension.')
    parser.add_argument('--final_layer', type = int, default = 0, help='Size of final layer of the GCN. [-2, -1, 0, integer > 0]. O : output dimension = input dimension. -1 : output dimension = input dimension//2. -2 : output dimension = input dimension//2.')
    parser.add_argument('--non_linearity', type = str, default= "none", help='Nonlinearity. [relu, tanh, none]')
    ###
    parser.add_argument('--learning_rate', type = float, default = 0.999e-2, help = 'Learning rate. [positive floats]' )
    parser.add_argument('--decay_learning_rate', type = str2bool, default = True, help='True or False. Apply or not a decay learning rate. [true, false]')
    parser.add_argument('--num_of_iter', type = int, default = 10, help='Number of epochs. [integer > 0]')
    parser.add_argument("--save_iter", nargs="+", default=[], help='List of epochs to save. [List of integer > 0]')
    parser.add_argument('--batch_size', type = int, default = 8, help='Batch size. [integer > 0]')
    parser.add_argument('--partial_train', type = float, default = 0.9, help='Fraction of dataset to train on. [0 < float < 0.9]. If > 0.9 then = 0.9.')
    parser.add_argument('--num_of_run', type = int, default = 10, help='Number of run. [integer > 0]')
    ###
    parser.add_argument('--sampling_type',type = str, default='orthov2', help='How to sample points for Monte-Carlo Estimation. For sw4d and pw4d. [regular, basis, orthov2, dppv2, hamm]')
    parser.add_argument('--sampling_nb', default = 50, help='Number of points to sample. [integer > 0]')
    ###
    parser.add_argument('--write_loss', type = str2bool, default = True, help='True or False. Decide whether or not to write loss training of model. [true, false]')
    parser.add_argument('--write_latent_space', type = str2bool, default = True, help='True or False. Decide whether or not to write model weights. [true, false]')
    parser.add_argument('--write_weights', type = str2bool, default = True, help='True or False. Decide whether or not to write model weights. [true, false]')
    ###
    parser.add_argument('--device', default='1', help='Index of the target GPU. Specify \'-1\' to disable gpu support.')
    parser.add_argument('--grid_search', type = str2bool, default = True, help='True or False. Decide whether or not to process a grid search. [true, false]')
    parser.add_argument('--evaluation', type = str2bool, default = True, help='True or False. Decide whether or not to evaluate latent space (evalutation function depend on the task selected). If option --num_of_run > 1 average evaluation of these run is returned. [true, false]')
    ###
    args = parser.parse_args()
    device = '/cpu:0' if args.device == '-1' or args.device == '' else '/gpu:'+args.device
    parameters = {}
    # Task and Dataset
    parameters["task"] = [args.task]
    parameters["dataset"] = [args.dataset] 
    parameters["feature"] = [args.feature] 
    parameters["num_of_layer"] = [args.num_of_layer]
    # GCN and Learning parameters
    if args.task == "sw4d" or args.task == "pw4d":
        parameters["loss"] = [args.loss] 
        parameters["gcn"] = [args.gcn]
        #
        parameters["num_of_layer"] = [args.num_of_layer]
        parameters["hidden_layer"] = [args.hidden_layer]
        parameters["final_layer"] = [args.final_layer]
        parameters["non_linearity"] = [args.non_linearity]
        #
        parameters["learning_rate"] = [args.learning_rate]
        parameters["decay_learning_rate"] = [args.decay_learning_rate]
        parameters["num_of_iter"] = [args.num_of_iter]
        parameters["save_iter"] = [args.save_iter]
        parameters["batch_size"] = [args.batch_size]
        parameters["partial_train"] = [args.partial_train]
        parameters["num_of_run"] = [args.num_of_run]
        # Monte-Carlo échantillonnage
        parameters["sampling_type"] = [args.sampling_type]
        parameters["sampling_nb"] = [args.sampling_nb]
    # Save data
    parameters["write_loss"] = [args.write_loss]
    parameters["write_latent_space"] = [True if args.write_latent_space or args.grid_search  else False ] 
    parameters["write_weights"] =  [args.write_weights] 
    parameters["evaluation"] = [args.evaluation] 
    if args.grid_search :
        parameters["num_of_layer"] = [0,1,2,3]
        if args.task == "pw4d" or args.task == "sw4d":
            parameters["feature"] = ["degree"] #["features","degree","node_labels","graph_fuse"]
            parameters["loss"] =  ["NCA", "LMNN-3", "NCMML"] 
            parameters["final_layer"] = [0,-1]
            parameters["decay_learning_rate"] = [True,False]
            parameters["partial_train"] = [0.2]
            parameters["sampling_type"] = ["orthov2","basis","hamm","dppv2"]
            parameters["non_linearity"] = ["relu"]
    if args.task in process_data.available_tasks() and args.dataset in process_data.available_datasets():
        with tf.device(device):
            list_of_parameters = list(ParameterGrid(parameters))
            num_list_of_parameters = len(list_of_parameters)
            for parameter_id, parameter in tqdm(enumerate(list_of_parameters), unit= "param"):
                if parameter_id in [k for k in range(num_list_of_parameters)]:
                    first_run = True
                    for run_id in range(args.num_of_run):
                        parameter = process_data.update_path(parameter, args.dataset+"_"+args.task, NOW, parameter_id, run_id)
                        if args.task == "sw4d":
                            data = process_data.load_dataset(args.dataset, parameter["attributes"])
                            distance = compute_dataset_sdistance(parameter, data, "sw4d", parameter["partial_train"], run_id)
                        if args.task == "pw4d":
                            data = process_data.load_dataset(args.dataset, parameter["attributes"])
                            distance = compute_dataset_sdistance(parameter, data, "pw4d", parameter["partial_train"], run_id)
                        if args.task == "wwl":
                            data = process_data.load_dataset(args.dataset, parameter["attributes"], h = parameter["num_of_layer"])
                            isdiscrete = True if parameter["attributes"] == "node_labels" or parameter["attributes"] == "degree" else False
                            distance = compute_wasserstein_distance(data["features"],parameter, h = 0, sinkhorn=False, discrete= isdiscrete, sinkhorn_lambda=1e-2)[-1]
                        #------------------
                        if first_run :
                            first_run = False
                            created_parameters_folder = [parameter]
                    status = "_success"
                    if args.write_latent_space or args.write_loss or args.write_weights:
                        for p in created_parameters_folder:
                            os.system('mv '+p["parameters_main_path"]+' '+p["parameters_main_path"]+status)
    else :
        if args.task not in process_data.available_tasks():
            print('Unknown dataset %s'%args.task)
        if args.dataset in  process_data.available_datasets():
            print('Unknown dataset %s'%args.dataset)
        parser.print_help()
    print("Fin.")
