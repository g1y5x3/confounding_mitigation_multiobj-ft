#!/usr/bin/env python
# coding: utf-8
import sys
import wandb
import argparse
import numpy.matlib
import multiprocessing
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from mlconfound.plot import plot_null_dist, plot_graph
from mlconfound.stats import partial_confound_test

from statsmodels.formula.api import ols

from multiprocessing.pool import ThreadPool

from pymoo.core.problem import StarmapParallelization
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

from pymoo.visualization.scatter import Scatter
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.display.multi import MultiObjectiveOutput

from ga.fitness import MyProblem, MyCallback

# Just to eliminate the warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="GA-SVM experiments")

    parser.add_argument('-ngen', type=int, default=4, help="Number of generation")
    parser.add_argument('-seed', type=int, default=8, help="Number of generation")
    parser.add_argument('-pop', type=int, default=128, help='Population size')
    parser.add_argument('-perm', type=int, default=500, help='Permutation value')
    parser.add_argument('-thread', type=int, default=8, help='Number of threads')
    parser.add_argument('-group', type=str, default='experiment_train36_test25', help='Group name')    

    args = parser.parse_args()
    project_name = 'LOO Vowels GA-SVM RBF'
    wandb.init(project=project_name, name="test_25", config=args)

    DATA_Train   = sio.loadmat("training_data_36-subjects.mat")
    DATA_Test_25 = sio.loadmat("testing_data_25-subjects.mat")

    feature_train = DATA_Train['FEAT']
    labels        = DATA_Train['LABEL']
    sub_skinfold  = DATA_Train['SUBJECT_SKINFOLD']

    X_train_36  = [np.stack(feature_train[i][0], axis=0) for i in range(36)]
    Y_train_36 = [labels[i][0].flatten() for i in range(36)]
    C_train_36 = [sub_skinfold[i][0].mean(axis=1) for i in range(36)]
    
    X_train_36, Y_train_36, C_train_36 = np.concatenate(X_train_36, axis=0), np.concatenate(Y_train_36, axis=0), np.concatenate(C_train_36, axis=0)
    print(f"X_train {X_train_36.shape}")

    feature_test_25 = DATA_Test_25['FEAT']
    labels_test_25  = DATA_Test_25['LABEL']
    sub_id_test_25  = DATA_Test_25['SUBJECT_ID']

    X_test_25 = [np.stack(feature_test_25[i][0], axis=0) for i in range(25)]
    Y_test_25 = [labels_test_25[i][0].flatten() for i in range(25)]
    X_test_25, Y_test_25 = np.concatenate(X_test_25, axis=0), np.concatenate(Y_test_25, axis=0)
    print([sub_id_test_25[i][0][0][0] for i in range(25)])

    config = {"num_generation"  : args.ngen,
              "population_size" : args.pop,
              "permutation"     : args.perm,
              "threads"         : args.thread}

    group_name = args.group

    X_means = np.load('X_means.npy')
    X_stds = np.load('X_stds.npy')
    print(X_means)
    print(X_stds)

    X_train = (X_train_36 - X_means[np.newaxis, :]) / X_stds[np.newaxis, :]
    Y_train = Y_train_36
    C_train = C_train_36
    
    X_test  = (X_test_25- X_means[np.newaxis, :]) / X_stds[np.newaxis, :]
    Y_test  = Y_test_25

    np.random.seed(10)
    shuffled_indices = np.random.permutation(len(X_train))

    X_train = X_train[shuffled_indices]
    Y_train = Y_train[shuffled_indices]
    C_train = C_train[shuffled_indices]
    print(X_train.shape)
    print(Y_train.shape)
    print(C_train.shape)

    clf = SVC(kernel='rbf')
    clf.fit(X_train, Y_train)
    label_train_predict = clf.predict(X_train)
    print("Training Acc:", accuracy_score(label_train_predict, Y_train))

    label_test_predict = clf.predict(X_test)
    print(f"Testing Acc: {accuracy_score(label_test_predict, Y_test):.2f}")

    ret = partial_confound_test(Y_train, label_train_predict, C_train, 
                                cat_y=True, cat_yhat=True, cat_c=False,
                                cond_dist_method='gam',
                                progress=False)
    print("P Value:", ret.p)

    wandb.log({"metrics/train_acc" : accuracy_score(label_train_predict, Y_train),
               "metrics/test_acc"  : accuracy_score(label_test_predict, Y_test),
               "metrics/p_value"   : ret.p})

    print('Genetic Algorithm Optimization...')
    num_permu       = config["permutation"]
    num_generation  = config["num_generation"]
    population_size = config["population_size"]
    threads_count   = config["threads"]

    n_threads = threads_count
    pool = ThreadPool(n_threads)
    runner = StarmapParallelization(pool.starmap)

    problem = MyProblem(elementwise_runner=runner)
    problem.load_data_svm(X_train, Y_train, C_train, clf, num_permu)

    # Genetic algorithm initialization
    algorithm = NSGA2(pop_size  = population_size,
                      sampling  = FloatRandomSampling(),
                      crossover = SBX(eta=15, prob=0.8),
                      mutation  = PM(eta=25),
                      output    = MultiObjectiveOutput())

    res = minimize(problem,
                   algorithm,
                   ("n_gen", num_generation),
                   callback = MyCallback(),
                   seed=args.seed,
                   verbose=False)

    print('Threads:', res.exec_time)
    pool.close()

    # Plot the parento front
    plt.figure()
    plt.scatter(res.F[:,0], res.F[:,1], marker='o', 
                                        edgecolors='red', 
                                        facecolor='None' )
    plt.xlabel("1-train_acc")
    plt.ylabel("1-p value")
    wandb.log({"plots/scatter_plot": wandb.Image(plt)})

    # Log and save the weights
    fw_dataframe = pd.DataFrame(res.X)
    fw_table = wandb.Table(dataframe=fw_dataframe)
    wandb.log({"feature weights": fw_table})

    # Evaluate the results discovered by GA
    Xid = np.argsort(res.F[:,0])
    test_acc_best = 0
    train_acc = 0
    p_value = 0
    for t in range(np.shape(res.X)[0]):
        w = res.X[Xid[t],:]

        # Evalute the training performance
        n = np.shape(X_train)[0]
        fw = np.matlib.repmat(w, n, 1)
        x_train_tf = X_train * fw
        Y_tf_train = clf.predict(x_train_tf)
        # temp_tr_acc = accuracy_score(Y_tf_train, Y_Train)
        temp_tr_acc = clf.score(x_train_tf, Y_train)
        # Evaluate the p value from the current predicitons
        ret_ga = partial_confound_test(Y_train, Y_tf_train, C_train, 
                                       cat_y=True, cat_yhat=True, cat_c=False,
                                       cond_dist_method='gam',
                                       progress=False)
        temp_p_value = ret_ga.p
        # Evaluate the testing performance
        n = np.shape(X_test)[0]
        fw = np.matlib.repmat(w, n, 1)
        x_test_tf = X_test * fw
        Y_tf_test = clf.predict(x_test_tf)
        temp_te_acc = accuracy_score(Y_tf_test, Y_test)

        wandb.log({"pareto-front/train_acc": temp_tr_acc,
                   "pareto-front/p_value"  : temp_p_value,
                   "pareto-front/test_acc" : temp_te_acc})

        print("\n")
        print("Training Acc after GA: ", temp_tr_acc)
        print("P Value      after GA: ", temp_p_value)
        print("Testing Acc after GA: ", temp_te_acc)

        if temp_te_acc > test_acc_best:
            test_acc_best = temp_te_acc
            train_acc = temp_tr_acc
            p_value = temp_p_value

    wandb.log({"metrics/train_acc_ga" : train_acc,
               "metrics/test_acc_ga"  : test_acc_best,
               "metrics/p_value_ga"   : p_value})

    wandb.finish()
