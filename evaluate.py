import argparse
import numpy as np
import pandas as pd
import glob
import os
import scipy.io as sio
from utils import process_data
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score ,normalized_mutual_info_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.model_selection import ParameterGrid, StratifiedKFold, KFold
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection._validation import _fit_and_score
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from sklearn.neighbors import KNeighborsClassifier
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt

def available_tasks():
    """ Return list of available tasks. """
    return ["sw4d",'wwl',"pw4d",'fgw']

def clean_parameters(parameters):
    new_parameters = {}
    keys_ = [key_ for key_ in parameters]
    for key_ in keys_:
        if "write" not in key_ and "__" not in key_:
            new_parameters[key_] = parameters[key_]
    return new_parameters

def rotate(list_, n):
    return list_[n:] + list_[:n]

def get_parameters_folder_path(search_path):
    return glob.glob(search_path+'/*success')

def get_run_folder_path(parameter_path, task, num_of_run):
    run_folder_path = glob.glob(parameter_path+'/*run*')
    parameter_file_path = glob.glob(parameter_path+'/*dict*')[0]
    if len(run_folder_path) == 1 and (task == "fgw" or task == "wwl"): #for fgw and wwl
        run_folder_path = run_folder_path*num_of_run
    return run_folder_path, parameter_file_path

def get_embedding_path(run_folder_path):
    #return [rotate(sorted(glob.glob(r+'/embeddings/*.mat')),1) for r in run_folder_path]
    return [rotate(sorted(glob.glob(r+'/embeddings/*.mat')),1)[0] for r in run_folder_path]

def embeddings_path2embeddings(embeddings_path):
    # return [[sio.loadmat(path)['distance'] for path in ep] for ep in embeddings_path]
    return [sio.loadmat(run_path)['distance'] for run_path in embeddings_path]

def custom_grid_search_cv(model, param_grid, precomputed_kernels, y, cv_num=5):
    '''
    Custom grid search based on the sklearn grid search for an array of precomputed kernels
    '''
    if cv_num  >= len(set(y)) :
        cv = StratifiedKFold(n_splits=cv_num , shuffle=False)
    else:
        cv = KFold(n_splits=cv_num , shuffle=False)
    results = []
    for train_index, test_index in cv.split(precomputed_kernels[0], y):
        split_results = []
        params = [] # list of dict, its the same for every split
        # run over the kernels first
        for K_idx, K in enumerate(precomputed_kernels):
            # Run over parameters
            for p in list(ParameterGrid(param_grid)):
                sc = _fit_and_score(clone(model), K, y, scorer=make_scorer(accuracy_score), 
                        train=train_index, test=test_index, verbose=0, parameters=p, fit_params=None)
                split_results.append(sc)
                params.append({'K_idx': K_idx, 'params': p})
        results.append(split_results)
    # Collect results and average
    results = np.array(results)
    fin_results = results.mean(axis=0)
    # select the best results
    best_idx = np.argmax(fin_results)
    # Return the fitted model and the best_parameters
    ret_model = clone(model).set_params(**params[best_idx]['params'])
    return ret_model.fit(precomputed_kernels[params[best_idx]['K_idx']], y), params[best_idx], fin_results[best_idx]

def evaluate_kernel_cv(embeddings_list,labels,list_of_parameters, list_of_run_path, search_path, cross_valid = 10 , cross_valid_max = 1):
    list_of_parameters = [a for a,e in zip(list_of_parameters,embeddings_list) if not np.isnan(np.sum(e))]
    list_of_run_path = [a for a,e in zip(list_of_run_path,embeddings_list) if not np.isnan(np.sum(e))]
    embeddings_list = [e for e in embeddings_list if not np.isnan(np.sum(e))]

    cross_valid = cross_valid
    cross_valid_max = cross_valid_max
    gammas = np.logspace(-4,1,num=6)  
    nb_parameters = len(list_of_parameters)
    nb_run = np.shape(embeddings_list)[1] 
    svm_valid_acc_results = np.zeros((cross_valid_max ,nb_parameters,nb_run)) 
    svm_test_acc_results = np.zeros((cross_valid_max ,nb_parameters,nb_run))
    labels = np.array([int(l) for l in labels])
    kernels  = [ [ [np.exp(-g*run) for g in gammas] for run in parameters] for parameters in embeddings_list]
    classif_tuned_parameters = {'C': [1]+list(np.logspace(-4,5,12))}#list(np.logspace(-3,3,8))}
    if cross_valid  >= len(set(labels)) :
        cv = StratifiedKFold(n_splits=cross_valid , shuffle=True)
    else:
        cv = KFold(n_splits=cross_valid , shuffle=True)
    param_number_str_list = [list_of_run_path[p][0].split('parameters')[-1].split("_success")[0] for p in range(nb_parameters)]
    for p in range(nb_parameters):
        addseed = 0
        for r in range(nb_run):
            if cross_valid_max == 1 :
                np.random.seed(42 + addseed)
            kernels_sublist = kernels[p][r]
            c = 0
            best_C = []
            best_gamma = []
            for train_index, test_index in cv.split(kernels_sublist[0], labels):
                if c < cross_valid_max :
                    K_train = [K[train_index][:, train_index] for K in kernels_sublist]
                    K_test  = [K[test_index][:, train_index] for K in kernels_sublist]
                    y_train, y_test = labels[train_index], labels[test_index]
                    gs, best_params, val_results = custom_grid_search_cv(SVC(kernel='precomputed'), 
                            classif_tuned_parameters, K_train, y_train, cv_num=5)
                    # Store best params
                    best_C.append( best_params['params']['C'] )
                    best_gamma.append( gammas[best_params['K_idx']])
                    y_pred = gs.predict(K_test[best_params['K_idx']])
                    svm_valid_acc_results[c,p,r] = val_results
                    svm_test_acc_results[c,p,r] = accuracy_score(y_test, y_pred)
                    c = c + 1
            columns0 = [ title for title in list_of_parameters[p]]
            list_of_parameters[p]['save_iter'] = [10]#:
            data_columns0 = [[list_of_parameters[p][c][0]]*cross_valid_max for c in columns0 ]
            columns = columns0 + ['C','gamma','svm_valid_mean','svm_test_mean']
            data_columns = data_columns0 + [ best_C, best_gamma, list(svm_valid_acc_results[:,p,r]), list(svm_test_acc_results[:,p,r])]
            index = ['fold_id{}'.format(i) for i in range(cross_valid_max)]
            pd.DataFrame(np.array(data_columns).T, 
            columns=columns, 
            index=index).to_csv(list_of_run_path[p][r]+"/evaluation/evaluation_kernel.csv")
            if cross_valid_max == 1 :
                    addseed += 1
        pd.DataFrame(np.array([[list_of_parameters[p][c][0]] for c in columns0 ]+[[np.mean(svm_valid_acc_results[:,p,:])],[np.mean(svm_test_acc_results[:,p,:])],[np.std(svm_test_acc_results[:,p,:])]]).T, 
                columns=columns0+['svm_valid_mean','svm_test_mean','svm_test_std'] ,
                index=['parameters '+param_number_str_list[p]]).to_csv(list_of_run_path[p][r].split('run')[0]+'parameters_evaluation_kernel.csv')
    #---------------
    arr = np.concatenate([np.array([[list_of_parameters[p][c][0]] for c in columns0 ]+[[np.mean(svm_valid_acc_results[:,p,:])],[np.mean(svm_test_acc_results[:,p,:])],[np.std(svm_test_acc_results[:,p,:])]]).T for p in range(len(list_of_parameters))] , axis = 0 )
    pd.DataFrame(arr, 
            columns=columns0+['svm_valid_mean','svm_test_mean','svm_test_std'] ,
            index=['parameters '+param_number_str_list[p] for p in range(len(list_of_parameters))]).to_csv(search_path+'/parameters_evaluation_kernel.csv')
    return np.mean(svm_valid_acc_results,axis = (0,1)), np.mean(svm_test_acc_results,axis = (0,1))

def evaluate_knn_cv(embeddings_list,labels,list_of_parameters, list_of_run_path, search_path, cross_valid = 10 , cross_valid_max = 1):
    list_of_parameters = [a for a,e in zip(list_of_parameters,embeddings_list) if not np.isnan(np.sum(e))]
    list_of_run_path = [a for a,e in zip(list_of_run_path,embeddings_list) if not np.isnan(np.sum(e))]
    embeddings_list = [e for e in embeddings_list if not np.isnan(np.sum(e))]

    cross_valid = cross_valid
    cross_valid_max = cross_valid_max
    nb_parameters = len(list_of_parameters)
    nb_run = np.shape(embeddings_list)[1] 
    knn_valid_acc_results = np.zeros((cross_valid_max ,nb_parameters,nb_run)) 
    knn_test_acc_results = np.zeros((cross_valid_max ,nb_parameters,nb_run))
    labels = np.array([int(l) for l in labels])
    distances  = [ [ [run] for run in parameters] for parameters in embeddings_list]
    classif_tuned_parameters = {'n_neighbors': np.array([1,2,3,5,7])}
    if cross_valid  >= len(set(labels)) :
        cv = StratifiedKFold(n_splits=cross_valid , shuffle=True)
    else:
        cv = KFold(n_splits=cross_valid , shuffle=True)
    param_number_str_list = [list_of_run_path[p][0].split('parameters')[-1].split("_success")[0] for p in range(nb_parameters)]
    for p in range(nb_parameters):
        addseed = 0
        for r in range(nb_run):
            if cross_valid_max == 1 :
                np.random.seed(42 + addseed)
            distances_sublist = distances[p][r]
            c = 0
            best_k = []
            best_null = []
            for train_index, test_index in cv.split(distances_sublist[0], labels):
                if c < cross_valid_max :
                    D_train = [D[train_index][:, train_index] for D in distances_sublist]
                    D_test  = [D[test_index][:, train_index] for D in distances_sublist]
                    y_train, y_test = labels[train_index], labels[test_index]
                    gs, best_params, val_results = custom_grid_search_cv(KNeighborsClassifier(metric='precomputed'), 
                            classif_tuned_parameters, D_train, y_train, cv_num=5)
                    best_k.append( best_params['params']['n_neighbors'] )
                    best_null.append(best_params['K_idx'])
                    y_pred = gs.predict(D_test[best_params['K_idx']])
                    knn_valid_acc_results[c,p,r] = val_results
                    knn_test_acc_results[c,p,r] = accuracy_score(y_test, y_pred)
                    c = c + 1
            columns0 = [ title for title in list_of_parameters[p]]
            list_of_parameters[p]['save_iter'] = [10]#:
            data_columns0 = [[list_of_parameters[p][c][0]]*cross_valid_max for c in columns0 ]
            columns = columns0 + ['k','null','knn_valid_mean','knn_test_mean']
            data_columns = data_columns0 + [ best_k, best_null, list(knn_valid_acc_results[:,p,r]), list(knn_test_acc_results[:,p,r])]
            index = ['fold_id{}'.format(i) for i in range(cross_valid_max)]
            pd.DataFrame(np.array(data_columns).T, 
            columns=columns, 
            index=index).to_csv(list_of_run_path[p][r]+"/evaluation/evaluation_knn.csv")
            if cross_valid_max == 1 :
                    addseed += 1
        pd.DataFrame(np.array([[list_of_parameters[p][c][0]] for c in columns0 ]+[[np.mean(knn_valid_acc_results[:,p,:])],[np.mean(knn_test_acc_results[:,p,:])],[np.std(knn_test_acc_results[:,p,:])]]).T, 
                columns=columns0+['knn_valid_mean','knn_test_mean','knn_test_std'] ,
                index=['parameters '+param_number_str_list[p]]).to_csv(list_of_run_path[p][r].split('run')[0]+'parameters_evaluation_knn.csv')
    #---------------
    arr = np.concatenate([np.array([[list_of_parameters[p][c][0]] for c in columns0 ]+[[np.mean(knn_valid_acc_results[:,p,:])],[np.mean(knn_test_acc_results[:,p,:])],[np.std(knn_test_acc_results[:,p,:])],[str(list(np.round(knn_test_acc_results[:,p,:][0,:],4)))]]).T for p in range(len(list_of_parameters))] , axis = 0 )
    # arr = np.concatenate([np.array([[list_of_parameters[p][c][0]] for c in columns0 ]+[[np.mean(knn_valid_acc_results[:,p,:])],[np.mean(knn_test_acc_results[:,p,:])],[np.std(knn_test_acc_results[:,p,:])]]).T for p in range(len(list_of_parameters))] , axis = 0 )
    pd.DataFrame(arr, 
            columns=columns0+['knn_valid_mean','knn_test_mean','knn_test_std', 'acc_list_test'] ,
            index=['parameters '+param_number_str_list[p] for p in range(len(list_of_parameters))]).to_csv(search_path+'/parameters_evaluation_knn.csv')
    return np.mean(knn_valid_acc_results,axis = (0,1)), np.mean(knn_test_acc_results,axis = (0,1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='pw4d', help='Task to execute. Only %s are currently available.'%str(available_tasks()))
    parser.add_argument('--dataset', default='MUTAG', help='Task to execute. Only %s are currently available.'%str(available_tasks()))
    parser.add_argument('--date', default='', help='[MONTH]_[DAY]_[YEAR]_[HOUR]h[MINUTES]m[SECONDES]s')
    parser.add_argument('--num_of_run', default=10, help='This parameter is required only for FGW and WWL. It set the number on which the results of the method are averaged.')
    
    args = parser.parse_args()
    if args.task in available_tasks():
        search_path_tab = ['./results/'+args.dataset+"_"+args.task+'/'+args.date]
        if args.date == '':
            search_path_tab = glob.glob(search_path_tab[0]+'/*')
        for search_path in search_path_tab:
            if os.path.isdir(search_path):
                list_of_parameters = [] 
                list_of_run_path = [] 
                embeddings_list  = []
                results = []
                parameters_folder_path = get_parameters_folder_path(search_path)
                for k in range(len(parameters_folder_path)):
                    run_folder_path, parameter_file_path = get_run_folder_path(parameters_folder_path[k], args.task, args.num_of_run)
                    embeddings_list.append( embeddings_path2embeddings(get_embedding_path(run_folder_path)) )
                    list_of_parameters.append( clean_parameters(sio.loadmat(parameter_file_path)) )
                    list_of_run_path.append(run_folder_path)
                labels = process_data.get_labels(args.dataset)

                for list_ in list_of_parameters:
                    list_copy = list_.copy()
                    for key in list_copy.keys():
                        try:
                            if 'path' in key :
                                del list_[key]
                        except KeyError:
                            pass
                # evaluate_kernel_cv(embeddings_list,labels,list_of_parameters, list_of_run_path, search_path)
                evaluate_knn_cv(embeddings_list,labels,list_of_parameters, list_of_run_path, search_path)
            else:
                print('Unknown date %s'%args.date)
                parser.print_help()
    else:
        print('Unknown task %s'%args.task)
        parser.print_help()
