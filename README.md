# SGML
# Simple Graph Metric Learning

## Requirements

- python 3.9.2
- scikit-learn 0.23.0, scikit-learn-extra 0.2.0
- numpy 1.19.5
- tensorflow 2.5
- python-igraph 0.9.6
- dppy 0.3.2 (https://github.com/guilgautier/DPPy)
- tqdm, argparse, matplotlib, typing

## Goal

The purpose of this repository is to make reproductible all experiences on "A simple way to learn metrics between graphs" paper [https://arxiv.org/abs/2209.12727]. These experiments consists in building a OT distance between graphs and use it to perform classification either with k-nn algorithm or SVM with a kernel built from the distance. The OT distance is learned from a differentiable GCN : SGCN [https://arxiv.org/pdf/1902.07153.pdf] and classical Metric Learning method/loss in literature : NCA [Goldberger et al, 2004, https://www.cs.toronto.edu/~hinton/absps/nca.pdf] and NCCML (inspired from NCMML [Mensink et al, 2012, https://hal.inria.fr/hal-00722313/document]) which is introduce in the paper. 

### Abstract 

    The choice of good distances and similarity measures between objects
    is important for many machine learning methods. Therefore, many
    metric learning algorithms have been developed in recent years, mainly for Euclidean 
    data in order to improve performance of classification or clustering methods. 
    However, due to difficulties in establishing computable, efficient and
    differentiable distances between attributed graphs, few metric learning algorithms 
    adapted to graphs have been developed despite the strong interest of the community.
    In this paper, we address this issue by proposing a new Simple Graph Metric 
    Learning - SGML - model with few trainable parameters based on Simple
    Convolutional Neural Networks - SGCN - and elements of optimal transport
    theory. This model allows us to build an appropriate distance from  a
    database of labeled (attributed) graphs to improve the performance of simple 
    classification algorithms such as $k$-NN. This distance can be quickly trained while 
    maintaining good performances as illustrated by the experimental study presented
    in this paper. 

## USAGE

### Dataset
All dataset in the paper can be found on :  https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
They must be placed in the "./data" folder.

### Command

    parser.add_argument('--task', default='pw4d', help='Task to execute. Only %s are currently available.'%str(process_data.available_tasks()))
    parser.add_argument('--dataset', default='NCI1', help='Dataset to work with. Only %s are currently available.'%str(process_data.available_datasets()))
    parser.add_argument('--feature', default = 'node_labels', help='Features to use for nodes. Only %s are currently available.'%str(process_data.available_tasks()))
    parser.add_argument('--loss', default = "NCCML", help='Metric learning loss')

    parser.add_argument('--gcn', type = str, default= "sgcn" , help='Type of GCN. [SGCN].')
    parser.add_argument('--num_of_layer', type = int, default= 2, help='Number of layer for GCN. [-1, integers > 0]. -1 : exponentiate. 0 : GCN = identity.')
    parser.add_argument('--hidden_layer_dim', type = int, default = 0, help='Size of hidden layer of the GCN if applicable. [integer > 0]. O : output dimension = input dimension.')
    parser.add_argument('--final_layer_dim', type = int, default = 0, help='Size of final layer of the GCN. [-2, -1, 0, integer > 0]. O : output dimension = input dimension. -1 : output dimension = input dimension//2. -2 : output dimension = input dimension//2.')
    parser.add_argument('--nonlinearity', type = str, default= "none", help='Nonlinearity. [relu, tanh, none]')

    parser.add_argument('--learning_rate', type = float, default = 0.999e-2, help = 'Learning rate. [positive floats]' )
    parser.add_argument('--decay_learning_rate', type = str2bool, default = True, help='True or False. Apply or not a decay learning rate. [true, false]')
    parser.add_argument('--num_of_iter', type = int, default = 10, help='Number of epochs. [integer > 0]')
    parser.add_argument("--save_iter", nargs="+", default=[10], help='List of epochs to save. [List of integer > 0]')
    parser.add_argument('--batch_size', type = int, default = 8, help='Batch size. [integer > 0]')
    parser.add_argument('--partial_train', type = float, default = 0.9, help='Fraction of dataset to train on. [0 < float < 0.9]. If > 0.9 then = 0.9.')
    parser.add_argument('--num_of_run', type = int, default = 10, help='Number of run. [integer > 0]')

    parser.add_argument('--sampling_type',type = str, default='ortho', help='How to sample points for Monte-Carlo Estimation. For sw4d and pw4d. [uniform, basis, orthov2, dppv2, hamm]')
    parser.add_argument('--sampling_nb', default = 50, help='Number of points to sample. [integer > 0]')

    parser.add_argument('--write_loss', type = str2bool, default = True, help='True or False. Decide whether or not to write loss training of model. [true, false]')
    parser.add_argument('--write_latent_space', type = str2bool, default = True, help='True or False. Decide whether or not to write model weights. [true, false]')
    parser.add_argument('--write_weights', type = str2bool, default = True, help='True or False. Decide whether or not to write model weights. [true, false]')

    parser.add_argument('--device', default='0', help='Index of the target GPU. Specify \'-1\' to disable gpu support.')
    parser.add_argument('--grid_search', type = str2bool, default = True, help='True or False. Decide whether or not to process a grid search. [true, false]')
    parser.add_argument('--evaluation', type = str2bool, default = True, help='True or False. Decide whether or not to evaluate latent space (evalutation function depend on the task selected). If option --num_of_run > 1 average evaluation of these run is returned. [true, false]')


### Reproduce paper runnding time comparison 

##### Command 

    python3 runtimes_evalutation.py 

It will save a "runtimes.mat" file 6 X 18 on folder "./" containing running times (in seconds) for different OT distance

column corrsponds to distribution size: (some size are not evaluated for all methods - which corresponds to value 0 in "runtimes.mat"-)
[      10,       22,       50,      114,      258,      581,
 1311,     2955,     6660,    15013,    33838,    76269,
171907,   387467,   873326,  1968419,  4436687, 10000000]

rows corresponds to method:

Wassestein 
Wassesertein entropic (100 reg parameters)
Sliced Wassesertein
Projected Wasserstein quadratic implementation (see the paper)
Projected Wasserstein sequential ipplementation 

### Reproduce paper classification experiments

#### MUTAG (with node labels) with Projected (Restricted) Sliced Wasserstein 

##### Training command

    python3 run.py -task psw4d  --grid_search True

Computed distances are written in results folder.
##### Evaluate command (custom kernel SVM and k-nn)
 
    python3 evaluate.py -task psw4d --dataset MUTAG

Results files can be found on results folder.

./results/DATASET_METHOD/TIMESTAMP/(*).csv for gird search global results

./results/DATASET_METHOD/TIMESTAMP/parameters(*)_sucess/run(*)/evaluation/(*).csv for specific (parameters sets,run) results

The grid search parameters can be set between line 260 and 288 of run.py file. The parameters of the search grid have priority over others (when grid_search is set to True).

Note: For PROTEINS you must specify wheter you wants to use concatenation of node labels and continuous features or only node labels. i.e:

    python3 evaluate.py -task psw4d --dataset PROTEINS --feature fuse

or

    python3 evaluate.py -task psw4d --dataset PROTEINS --feature node_labels

### Specific Example

#### MUTAG (with node labels) with Projected Restricted Sliced Wasserstein

##### Training command (with NCA loss)

    python3 run.py -task pw4d --dataset MUTAG -feature node_labels -loss NCA -gcn sgcn  --num_of_run 1 -sampling_type basis

##### Evaluate command (custom kernel SVM and k-nn)
 
    python3 evaluate.py -task pw4d --dataset MUTAG


#### COX2 (with "continuous" features) with Projected Sliced Wasserstein

##### Training command (with NCCML loss)

    python3 run.py -task pw4d --dataset COX2 -feature attributes -loss NCCML -gcn sgcn --grid_search' --num_of_run 1 -sampling_type uniform

##### Evaluate command (custom kernel SVM and k-nn)
 
    python3 evaluate.py -task pw4d --dataset COX2

#### BZR (with "continuous" features concatened with nodel_labels) with Sliced Wasserstein

##### Training command (with NCCML loss)

    python3 run.py -task sw4d --dataset BZR -feature fuse -loss NCCML -gcn sgcn --grid_search' --num_of_run 1 -sampling_type uniform

##### Evaluate command (custom kernel SVM and k-nn)
 
    python3 evaluate.py -task sw4d --dataset BZR

#### BZR (with degree as features) with Sliced Wasserstein (with dpp process sampling)

##### Training command (with NCCML loss)

    python3 run.py -task sw4d --dataset BZR -feature fuse -loss NCCML -gcn sgcn --grid_search' --num_of_run 1 -sampling_type dppv2

##### Evaluate command (custom kernel SVM and k-nn)
 
    python3 evaluate.py -task sw4d --dataset BZR

WWL and a sliced version SWWL can be also run in the same way, however there is no adaptation since the features are generated for these models by a non trainable GCN (Weisfeiler-Lheman features).
