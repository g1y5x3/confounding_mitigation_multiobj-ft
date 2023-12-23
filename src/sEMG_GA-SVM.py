#!/usr/bin/env python
# coding: utf-8
import os
import sys
import wandb
import argparse
import numpy as np
import scipy.io as sio

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from mlconfound.stats import partial_confound_test
from multiprocessing.pool import ThreadPool

from pymoo.optimize import minimize
from pymoo.core.problem import StarmapParallelization
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.display.multi import MultiObjectiveOutput

from util.fitness import MyProblem, MyCallback

# Just to eliminate the warnings
def warn(*args, **kwargs):
  pass
import warnings
warnings.warn = warn

WANDB = os.getenv("WANDB", False)
GROUP = os.getenv("GROUP", "tests")

if __name__ == "__main__":

  DATA_ALL = sio.loadmat("data/subjects_40_v6.mat")
  FEAT_N           = DATA_ALL['FEAT_N']            # Normalized features
  LABEL            = DATA_ALL['LABEL']             # Labels
  VOWEL            = DATA_ALL['LABEL_VOWEL']       # Type of Vowels
  VFI_1            = DATA_ALL['SUBJECT_VFI']       # VFI-1 Score
  SUBJECT_ID       = DATA_ALL['SUBJECT_ID']        # Sujbect ID
  SUBJECT_SKINFOLD = DATA_ALL['SUBJECT_SKINFOLD']  # Subject Skinfold Thickness

  leftout = 1

  training_acc = np.zeros(40)
  testing_acc  = np.zeros(40)
  p_value      = np.zeros(40)

  testing_acc_ga  = np.zeros(40)
  training_acc_ga = np.zeros(40)
  p_value_ga      = np.zeros(40)

  project_name = 'Confounding-Mitigation-In-Deep-Learning'

  parser = argparse.ArgumentParser(description="GA-SVM experiments")
  parser.add_argument('-s',      type=int, default=0,  help="start of the subjects")
  parser.add_argument('-nsub',   type=int, default=1,  help="number of subjects to be executed")
  parser.add_argument('-ngen',   type=int, default=5,  help="Number of generation")
  parser.add_argument('-pop',    type=int, default=8,  help='Population size')
  parser.add_argument('-perm',   type=int, default=10, help='Permutation value')
  parser.add_argument('-thread', type=int, default=8,  help='Number of threads')
  args = parser.parse_args()
 
  # it controls which subject to start the Leave-One-Out exp and how many loops to run
  start_sub  = args.s 
  num_sub    = args.nsub
  for sub_test in range(start_sub, start_sub + num_sub): 

    sub_txt = "R%03d"%(int(SUBJECT_ID[sub_test][0][0]))

    if int(VFI_1[sub_test][0][0]) > 10:
      sub_group = 'Fatigued'
    else:
      sub_group = 'Healthy'

    # Default value for configurations and parameters that doesn't need
    # to be logged
    config = {"num_generation"  : args.ngen,
              "population_size" : args.pop,
              "permutation"     : args.perm,
              "threads"         : args.thread}

    if WANDB:
      run = wandb.init(project = project_name,
                      group    = GROUP,
                      config   = config,
                      name     = sub_txt,
                      tags     = [sub_group],  # for convenience later to visualize results grouped by `healthy` or `fatigued`
                      settings = wandb.Settings(_disable_stats=True, _disable_meta=True),
                      reinit   = True)

    print('\n===No.%d: %s===\n'%(sub_test+1, sub_txt)) 
    print('VFI-1:', (VFI_1[sub_test][0][0]))


    # ===== Load Testing Samples =====
    num_signal = np.shape(FEAT_N[sub_test,0])[0]    
    X_Temp = FEAT_N[sub_test,0]
    Y_Temp = LABEL[sub_test,0].flatten()

    num_leftout = round(leftout*num_signal)
    index_leftout = np.random.choice(range(num_signal), 
                                        size=num_leftout, 
                                        replace=False)
    print("Left-out Test samples: ", index_leftout.size)

    X_Test = X_Temp[index_leftout,:]
    Y_Test = Y_Temp[index_leftout]

    index_include = np.arange(num_signal)
    index_include = np.delete(index_include, index_leftout)
    print("Included Training samples: ", index_include.size)
    X_include = X_Temp[index_include,:]
    Y_include = Y_Temp[index_include]

    # ===== Load Training Samples =====
    X = np.zeros((0,48))
    Y = np.zeros(0)    
    C = np.zeros(0)
    for sub_train in range(40):
      if sub_train != sub_test:
        x_s = FEAT_N[sub_train,0]
        y_s = LABEL[sub_train,0].flatten()
        c_s = np.mean(np.mean(SUBJECT_SKINFOLD[sub_train,:]), axis=1)
        X = np.concatenate((X, x_s), axis=0)
        Y = np.concatenate((Y, y_s), axis=0)
        C = np.concatenate((C, c_s), axis=0)       

    print('# of Healthy Samples: %d'%(np.sum(Y == -1)))
    print('# of Fatigued Samples: %d'%(np.sum(Y == 1)))    

    # ===== Training and Validation =====
    # NOTE the experiment doesn't really use the validation set
    X_Train, _, YC_Train, _ = train_test_split(X, np.transpose([Y, C]), test_size=0.1, random_state=42)
    Y_Train, C_Train = YC_Train[:,0], YC_Train[:,1]

    clf = SVC(C=1.0, gamma='scale', kernel='rbf', class_weight='balanced', max_iter=1000, tol=0.001)
    clf.fit(X_Train, Y_Train)

    label_predict = clf.predict(X_Train)

    print('Training Acc: ', accuracy_score(label_predict, Y_Train))
    training_acc[sub_test] = accuracy_score(label_predict, Y_Train)

    ret = partial_confound_test(Y_Train, label_predict, C_Train, 
                                cat_y=True, cat_yhat=True, cat_c=False,
                                cond_dist_method='gam',
                                progress=False)
    p_value[sub_test] = ret.p
    print('P Value     : ', p_value[sub_test])
    # TODO upload the CPT figure to to wandb as well

    label_predict = clf.predict(X_Test)
    print('Testing  Acc: ', accuracy_score(label_predict, Y_Test))
    testing_acc[sub_test] = accuracy_score(label_predict, Y_Test)

    if WANDB:
      wandb.log({"subject_info/vfi_1" : int(VFI_1[sub_test][0][0]),
                 "metrics/train_acc"  : training_acc[sub_test],
                 "metrics/test_acc"   : testing_acc[sub_test],
                 "metrics/p_value"    : p_value[sub_test]})

    print("Genetic Algorithm Optimization...")

    num_permu       = config["permutation"]
    num_generation  = config["num_generation"]
    population_size = config["population_size"]
    threads_count   = config["threads"]

    n_threads = threads_count
    pool = ThreadPool(n_threads)
    runner = StarmapParallelization(pool.starmap)

    problem = MyProblem(elementwise_runner=runner)
    problem.load_param(X_Train, Y_Train, C_Train, X_Test, Y_Test, 
                       clf, num_permu)

    # Genetic algorithm initialization
    algorithm = NSGA2(pop_size  = population_size,
                      sampling  = FloatRandomSampling(),
                      crossover = SBX(eta=15, prob=0.9),
                      mutation  = PM(eta=20),
                      output    = MultiObjectiveOutput())

    res = minimize(problem,
                   algorithm,
                   ("n_gen", num_generation),
                   callback = MyCallback(),
                   verbose=False)

    print('Threads:', res.exec_time)
    pool.close()
    training_acc_ga[sub_test] = res.algorithm.callback.data["train_acc"][-1]
    p_value_ga[sub_test] = res.algorithm.callback.data["p_value"][-1]
    testing_acc_ga[sub_test] = res.algorithm.callback.data["test_acc"][-1]

    print("Training Acc after GA: ", training_acc_ga[sub_test])
    print("P Value      after GA: ", p_value_ga[sub_test])
    print("Testing  Acc after GA: ", testing_acc_ga[sub_test])

    if WANDB:
      wandb.log({"metrics/train_acc_ga" : training_acc_ga[sub_test],
                 "metrics/test_acc_ga"  : testing_acc_ga[sub_test],
                 "metrics/p_value_ga"   : p_value_ga[sub_test]})
      run.finish()
