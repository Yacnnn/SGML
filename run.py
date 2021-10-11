import os 
import argparse
import ot
import numpy as np
import scipy.io as sio
import tensorflow as tf

from datetime import datetime
from tqdm import tqdm

from utils import process
from utils import process_data

from sklearn.model_selection import ParameterGrid, StratifiedKFold

from models.sw4d import Sw4d
from models.pw4d import Pw4d

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
   
def create_batch(features, structures, labels, batch_size = 32, shuffle = True, precomputed_batch = False):
    """ Creates a list of batch of data. """
    X = np.copy(features)
    W = np.copy(structures)
    labels = np.copy(labels)
    n = X.shape[0]
    if shuffle:
        s = np.arange(n) 
        np.random.shuffle(s)
        X = features[s]  
        W = W[s]
        labels = labels[s]
    q = n//batch_size
    block_end = q*batch_size   
    batch_limit = [ [k*batch_size,(k+1)*batch_size]  for k in range(q)] + [[q*batch_size,n]]
    if batch_limit[-1][1]  - batch_limit[-1][0]  == 1 :
        batch_limit[-2][1] =  batch_limit[-1][1]
        batch_limit = batch_limit[:-1]
    batch_features = [ X[ind[0]:ind[1]] for ind in batch_limit]
    batch_structures = [ W[ind[0]:ind[1]] for ind in batch_limit]
    batch_labels = [ labels[ind[0]:ind[1]] for ind in batch_limit]
    batch_indice = [  s[ind[0]:ind[1]]  for ind in batch_limit]#.astype(np.float32)
    if batch_features[-1].shape[0] == 0 :
        return batch_features[:-1], batch_structures[:-1], batch_labels[:-1], batch_indice[:-1]
    return batch_features, batch_structures, batch_labels, batch_indice

def save_distance(distance, parameters, title_extension = ""):
    new_parameters = {}
    for key_ in parameters:
        if "write" not in key_:# and "path" not in key_:
            new_parameters[key_] = parameters[key_]
    sio.savemat(parameters["latent_space_path"]+"distance"+title_extension+".mat", mdict={
                    "distance" : distance, 
                    "parameters" : new_parameters })

def compute_dataset_distance(parameters, data, model_name = "sw4d", partial_train = 0.9, addseed = 0):
    """ Train the model given multiviews data and specified parameters and return it. """  
    cv = StratifiedKFold(n_splits = 10 , shuffle=True) 
    np.random.seed(42 + addseed)
    train_index, test_index = next(cv.split(data["features"], data["labels"]))
    if partial_train < 0.9 :
        keep_number = int(len(data["features"])*partial_train)
        s = np.copy(train_index)
        np.random.shuffle(s)
        s = np.sort(s[:keep_number])
        features = data["features"][s]
        structures = data["structures"][s]
        labels = data["labels"][s]
    else :
        features = data["features"][train_index]
        structures = data["structures"][train_index]
        labels = data["labels"][train_index]
    # Model parameters
    gcn = parameters["gcn"]
    loss_name = parameters["loss"]
    num_of_layer = parameters["num_of_layer"]
    hidden_layer_dim = parameters["hidden_layer_dim"]
    final_layer_dim = parameters["final_layer_dim"]
    sampling_nb = parameters["sampling_nb"]
    sampling_type = parameters["sampling_type"]
    nonlinearity = parameters["nonlinearity"]
    # Training parameters
    learning_rate = parameters["learning_rate"]
    decay_learning_rate = parameters["decay_learning_rate"]
    num_of_iter = parameters["num_of_iter"]
    save_iter = parameters["save_iter"]
    ###########
    dataset = parameters["dataset"]
    batch_size = parameters["batch_size"]
    ###########
    Xw4d = Sw4d if model_name == "sw4d" else Pw4d if model_name == "pw4d" else None
    model_xw4d = Xw4d( 
                        gcn_type = gcn,
                        l2reg = 0,
                        loss_name = loss_name,
                        num_of_layer = num_of_layer,
                        hidden_layer_dim = hidden_layer_dim,
                        final_layer_dim = final_layer_dim,
                        nonlinearity = nonlinearity,
                        sampling_type = sampling_type, 
                        num_of_theta_sampled = sampling_nb,
                        dataset = dataset
                    )
    # model_xw4d.build_transport_matrix(list(data["features"]))
    # old_loss_d = -1
    for e in tqdm(range(num_of_iter), unit= "iter" ):
        # print( "epochs : " + str(e) + "/" + str(num_of_iter))
        batch_features, batch_structures, batch_labels, batch_indice = create_batch(features, structures, labels, batch_size = batch_size, shuffle = True)
        optimizer = tf.keras.optimizers.Adam(lr = learning_rate)
        acc_loss_d = 0
        for feat, struct, lab, s in zip( tqdm( batch_features, unit = "batch", disable = True ), batch_structures, batch_labels, batch_indice ):
            with tf.GradientTape() as tape:
                loss_d, loss_s =  model_xw4d(list(feat),list(struct),list(lab),list(s))
            gradients = tape.gradient(loss_d, model_xw4d.trainable_variables)
            gradient_variables = zip( gradients, model_xw4d.trainable_variables)
            optimizer.apply_gradients(gradient_variables)
            acc_loss_d += loss_d/len(batch_indice)
        # if tf.abs( (old_loss_d - loss_d ) / loss_d ) < 1e-6 :
        #     break
        # old_loss_d = loss_d
        if decay_learning_rate :
            optimizer.learning_rate = learning_rate * np.math.pow(1.1, - 50.*(e / num_of_iter))
        print("epochs : " + str(e) + "/" + str(num_of_iter))
        # print("loss_d : ", str(loss_d) )
        print("avg_loss_d : ", str(acc_loss_d) )
        if e + 1  in save_iter  and e != num_of_iter - 1 :
            D = model_xw4d.distance_fastv2(list(data["features"]),list(data["structures"]))
            parameters_copy = parameters.copy()
            parameters_copy["num_of_iter"] = e + 1
            save_distance(distance = D.numpy(), parameters = parameters_copy , title_extension= "_iter"+str(e + 1))   
    D = model_xw4d.distance_fastv2(list(data["features"]),list(data["structures"]))
    save_distance(distance = D.numpy(), parameters = parameters, title_extension= "_iter"+str(e + 1))
    return D

# from https://github.com/BorgwardtLab/WWL
def compute_wasserstein_distance(label_sequences, parameters, h, sinkhorn=False, 
                                    discrete=False, sinkhorn_lambda=1e-2):
    '''
    Generate the Wasserstein distance matrix for the graphs embedded 
    in label_sequences
    '''
    # Get the iteration number from the embedding file
    n = len(label_sequences)
    emb_size = label_sequences[0].shape[1]
    n_feat = int(emb_size/(h+1))
    # Iterate over all possible h to generate the Wasserstein matrices
    # hs = range(0, h + 1)
    hs = range(h, h + 1)
    wasserstein_distances = []
    for h in hs:
        M = np.zeros((n,n))
        # Iterate over pairs of graphs
        for graph_index_1, graph_1 in enumerate(label_sequences):
            # Only keep the embeddings for the first h iterations
            labels_1 = label_sequences[graph_index_1][:,:n_feat*(h+1)]
            for graph_index_2, graph_2 in enumerate(label_sequences[graph_index_1:]):
                labels_2 = label_sequences[graph_index_2 + graph_index_1][:,:n_feat*(h+1)]
                # Get cost matrix
                ground_distance = 'hamming' if discrete else 'euclidean'
                costs = ot.dist(labels_1, labels_2, metric=ground_distance)

                if sinkhorn:
                    mat = ot.sinkhorn(np.ones(len(labels_1))/len(labels_1), 
                                        np.ones(len(labels_2))/len(labels_2), costs, sinkhorn_lambda, 
                                        numItermax=50)
                    M[graph_index_1, graph_index_2 + graph_index_1] = np.sum(np.multiply(mat, costs))
                else:
                    M[graph_index_1, graph_index_2 + graph_index_1] = \
                        ot.emd2([], [], costs) 
        M = (M + M.T)
        wasserstein_distances.append(M)
        print(f'Iteration {h}: done.')
    save_distance(wasserstein_distances[-1], parameters)
    return wasserstein_distances

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='sw4d', help='Task to execute. Only %s are currently available.'%str(process_data.available_tasks()))
    parser.add_argument('--dataset', default='DD', help='Dataset to work with. Only %s are currently available.'%str(process_data.available_datasets()))
    parser.add_argument('--feature', default = 'degree', help='Features to use for nodes. Only %s are currently available.'%str(process_data.available_tasks()))
    parser.add_argument('--loss', default = "NCCML", help='Metric learning loss')
    ###    
    parser.add_argument('--gcn', type = str, default= "sgcn" , help='Type of GCN. [SGCN].')
    parser.add_argument('--num_of_layer', type = int, default= 2, help='Number of layer for GCN. [-1, integers > 0]. -1 : exponentiate. 0 : GCN = identity.')
    parser.add_argument('--hidden_layer_dim', type = int, default = 0, help='Size of hidden layer of the GCN if applicable. [integer > 0]. O : output dimension = input dimension.')
    parser.add_argument('--final_layer_dim', type = int, default = 0, help='Size of final layer of the GCN. [-2, -1, 0, integer > 0]. O : output dimension = input dimension. -1 : output dimension = input dimension//2. -2 : output dimension = input dimension//2.')
    parser.add_argument('--nonlinearity', type = str, default= "none", help='Nonlinearity. [relu, tanh, none]')
    ###
    parser.add_argument('--learning_rate', type = float, default = 0.999e-2, help = 'Learning rate. [positive floats]' )
    parser.add_argument('--decay_learning_rate', type = str2bool, default = True, help='True or False. Apply or not a decay learning rate. [true, false]')
    parser.add_argument('--num_of_iter', type = int, default = 10, help='Number of epochs. [integer > 0]')
    parser.add_argument("--save_iter", nargs="+", default=[10], help='List of epochs to save. [List of integer > 0]')
    parser.add_argument('--batch_size', type = int, default = 8, help='Batch size. [integer > 0]')
    parser.add_argument('--partial_train', type = float, default = 0.9, help='Fraction of dataset to train on. [0 < float < 0.9]. If > 0.9 then = 0.9.')
    parser.add_argument('--num_of_run', type = int, default = 10, help='Number of run. [integer > 0]')
    ###
    parser.add_argument('--sampling_type',type = str, default='ortho', help='How to sample points for Monte-Carlo Estimation. For sw4d and pw4d. [uniform, basis, orthov2, dppv2, hamm]')
    parser.add_argument('--sampling_nb', default = 50, help='Number of points to sample. [integer > 0]')
    ###
    parser.add_argument('--write_loss', type = str2bool, default = True, help='True or False. Decide whether or not to write loss training of model. [true, false]')
    parser.add_argument('--write_latent_space', type = str2bool, default = True, help='True or False. Decide whether or not to write model weights. [true, false]')
    parser.add_argument('--write_weights', type = str2bool, default = True, help='True or False. Decide whether or not to write model weights. [true, false]')
    ###
    parser.add_argument('--device', default='0', help='Index of the target GPU. Specify \'-1\' to disable gpu support.')
    parser.add_argument('--grid_search', type = str2bool, default = True, help='True or False. Decide whether or not to process a grid search. [true, false]')
    parser.add_argument('--evaluation', type = str2bool, default = True, help='True or False. Decide whether or not to evaluate latent space (evalutation function depend on the task selected). If option --num_of_run > 1 average evaluation of these run is returned. [true, false]')
    ###
    args = parser.parse_args()
    device = '/cpu:0' if args.device == '-1' or args.device == '' else '/gpu:'+args.device
    parameters = {}
    # Task and Dataset
    parameters["task"] = [args.task]
    parameters["dataset"] = [args.dataset] 
    parameters["features"] = [args.feature] 
    parameters["num_of_layer"] = [args.num_of_layer]
    # GCN and Learning parameters
    if args.task == "sw4d" or args.task == "pw4d":
        parameters["loss"] = [args.loss] 
        parameters["gcn"] = [args.gcn]
        #
        parameters["num_of_layer"] = [args.num_of_layer]
        parameters["hidden_layer_dim"] = [args.hidden_layer_dim]
        parameters["final_layer_dim"] = [args.final_layer_dim]
        parameters["nonlinearity"] = [args.nonlinearity]
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
        parameters["num_of_layer"] = [1,2,3,4]
        if args.task == "pw4d" or args.task == "sw4d":
            parameters["feature"] = ["degree"] #["features","degree","node_labels","graph_fuse"]
            parameters["loss"] =  ["NCCML"] #"NCA", "LMNN-3", 
            parameters["final_layer_dim"] = [5]
            parameters["decay_learning_rate"] = [False]
            parameters["partial_train"] = [0.9,0.2]
            parameters["sampling_type"] = ["ortho",'basis']
            parameters["nonlinearity"] = ["relu"]
            parameters["num_of_iter"] = [10]
            parameters["sampling_nb"] = [50] 
            parameters["batch_size"] = [8] 
    if args.task in process_data.available_tasks() and args.dataset in process_data.available_datasets():
        with tf.device(device):
            list_of_parameters = list(ParameterGrid(parameters))
            num_list_of_parameters = len(list_of_parameters)
            for parameter_id, parameters_ in tqdm(enumerate(list_of_parameters), unit= "param"):
                if parameter_id in [k for k in range(num_list_of_parameters)][0:2]:
                    first_run = True
                    for run_id in range(args.num_of_run):
                        parameters_ = process_data.update_path(parameters_, args.dataset+"_"+args.task, NOW, parameter_id, run_id)
                        if args.task == "sw4d":
                            data = process_data.load_dataset(args.dataset, parameters_["features"])
                            distance = compute_dataset_distance(parameters_, data, "sw4d", parameters_["partial_train"], run_id)
                        if args.task == "pw4d":
                            data = process_data.load_dataset(args.dataset, parameters_["features"])
                            distance = compute_dataset_distance(parameters_, data, "pw4d", parameters_["partial_train"], run_id)
                        if args.task == "wwl":
                            data = process_data.load_dataset(args.dataset, parameters_["features"], h = parameters_["num_of_layer"])
                            isdiscrete = True if parameters_["features"] == "node_labels" or parameters_["features"] == "degree" else False
                            distance = compute_wasserstein_distance(data["features"],parameters_, h = parameters_["num_of_layer"], sinkhorn=False, discrete= isdiscrete, sinkhorn_lambda=1e-2)[-1]
                        #------------------
                        if first_run :
                            first_run = False
                            created_parameters_folder = [parameters_]
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
