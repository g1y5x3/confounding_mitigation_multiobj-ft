#!/usr/bin/env python
# coding: utf-8
import argparse
import numpy.matlib
import multiprocessing
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt

from sklearn.svm import SVC
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

import wandb
from statsmodels.formula.api import ols
from pymoo.core.callback import Callback
from pymoo.core.problem import ElementwiseProblem

# from GA.fitness_log import MyProblem, MyCallback

class MyProblem(ElementwiseProblem):

    def __init__(self, **kwargs):
        super().__init__(n_var=48, 
                         n_obj=2,
                         n_constr=0,
                         xl = -2*np.ones(48),
                         xu =  2*np.ones(48),
                         **kwargs)

    def load_param(self, x_train, y_train, c_train, x_test, y_test, clf, permu):
        # Load informations from the individual classification exerpiment 
        # x_train - training features
        # y_train - labels
        # c_train - confounding variables
        # model   - the trained svm model
        self.x_train  = x_train
        self.y_train  = y_train
        self.c_train  = c_train
        self.x_test   = x_test
        self.y_test   = y_test
        self.clf      = clf
        self.permu    = permu

        # dimension of the training feature
        self.n = np.shape(x_train)[0]
        self.d = np.shape(x_train)[1]
 
    def _evaluate(self, x, out, *args, **kwargs):
        # pymoo initialize the chromosome as a 1-D array which can be converted
        # into matrix for element-wise weight multiplication
        fw = np.matlib.repmat(x, self.n, 1)
        x_train_tf = self.x_train * fw

        # first objective is SVM training accuracy
        f1 = 1 - self.clf.score(x_train_tf, self.y_train)

        # second objective is P Value from CPT  
        y_hat = self.clf.predict(x_train_tf)
        ret = partial_confound_test(self.y_train, y_hat, self.c_train,
                                    cat_y=True, cat_yhat=True, cat_c=False,
                                    cond_dist_method='gam', 
                                    num_perms=self.permu, mcmc_steps=50,
                                    n_jobs=-1,
                                    progress=False)

        f2 = 1 - ret.p 

        out['F'] = [f1, f2]

class MyCallback(Callback):
    def __init__(self, log_flag) -> None:
        super().__init__()
        self.data["train_acc"] = []
        self.data["test_acc"] = []
        self.data["p_value"] = []
        self.data["rsquare"] = []
        self.data["predict"] = []
        self.log_flag = log_flag

    def notify(self, algorithm):
        F = algorithm.pop.get("F")
        X = algorithm.pop.get("X")
        
        # Evaluate the results from GA
        Xid = np.argsort(F[:,0])
        acc_best = 0        
        tr_acc_best  = 0 
        p_value_best = 0 
        rsqrd_best   = 0 
        te_acc_best  = 0 
        predict_best = []
        for t in range(np.shape(X)[0]):
            w = X[Xid[t],:]

            # Evalute the training performance
            fw = np.matlib.repmat(w, algorithm.problem.n, 1)
            x_train_tf  = algorithm.problem.x_train * fw
            y_train_tf  = algorithm.problem.clf.predict(x_train_tf)
            temp_tr_acc = algorithm.problem.clf.score(x_train_tf, 
                                                      algorithm.problem.y_train)

            # Evaluate the r squared
            df = pd.DataFrame({'x': algorithm.problem.c_train, 'y': y_train_tf})
            fit = ols('y~C(x)', data=df).fit()
            temp_rsqrd = fit.rsquared.flatten()[0]

            # Evaluate the p value from the current predicitons
            ret_ga = partial_confound_test(algorithm.problem.y_train, 
                                           y_train_tf, 
                                           algorithm.problem.c_train, 
                                           cat_y=True, 
                                           cat_yhat=True, 
                                           cat_c=False,
                                           cond_dist_method='gam',
                                           progress=False)
            temp_p_value = ret_ga.p

            # Evaluate the testing performance
            n = np.shape(algorithm.problem.x_test)[0]
            fw = np.matlib.repmat(w, n, 1)
            x_test_tf = algorithm.problem.x_test * fw
            temp_te_acc = algorithm.problem.clf.score(x_test_tf, 
                                                      algorithm.problem.y_test)

            label_pred = algorithm.problem.clf.predict(x_test_tf)

            if self.log_flag:
                wandb.log({"pareto-front/train_acc-{}".format(algorithm.n_gen): temp_tr_acc,
                           "pareto-front/rsquare-{}".format(algorithm.n_gen)  : temp_rsqrd,
                           "pareto-front/p_value-{}".format(algorithm.n_gen)  : temp_p_value,
                           "pareto-front/test_acc-{}".format(algorithm.n_gen) : temp_te_acc})

            if temp_te_acc > acc_best:
                acc_best = temp_te_acc 

                tr_acc_best  = temp_tr_acc 
                p_value_best = temp_p_value
                rsqrd_best   = temp_rsqrd
                te_acc_best  = temp_te_acc
                predict_best = algorithm.problem.clf.predict(x_test_tf)

        if self.log_flag:
            wandb.log({"ga/n_gen"     : algorithm.n_gen,
                       "ga/train_acc" : tr_acc_best,
                       "ga/p_value"   : p_value_best,
                       "ga/rsquare"   : rsqrd_best,
                       "ga/test_acc"  : te_acc_best})

        self.data["train_acc"].append(tr_acc_best)
        self.data["p_value"].append(p_value_best)
        self.data["rsquare"].append(rsqrd_best)  
        self.data["test_acc"].append(te_acc_best)
        self.data["predict"].append(predict_best)

# Just to eliminate the warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import wandb

if __name__ == "__main__":

    project_name = 'LOO-Sentence-Classification'

    parser = argparse.ArgumentParser(description="GA-SVM experiments")

    parser.add_argument('-win',    type=str, default='fix')
    parser.add_argument('-wandb',  type=int, default=0)
    parser.add_argument('-sen',    type=int, default=1, help="the type of vowels that you want to investigate")
    parser.add_argument('-start',  type=int, default=0, help="start of the subjects")
    parser.add_argument('-nsub',   type=int, default=1, help="number of subjects to be executed")
    parser.add_argument('-ngen',   type=int, default=2, help="Number of generation")
    parser.add_argument('-pop',    type=int, default=16, help='Population size')
    parser.add_argument('-perm',   type=int, default=100, help='Permutation value')
    parser.add_argument('-thread', type=int, default=8, help='Number of threads')
    parser.add_argument('-group',  type=str, default='experiment_test', help='Group name')

    args = parser.parse_args()

    # Default value for configurations and parameters that doesn't need
    # to be logged
    config = {"num_generation"  : args.ngen,
              "population_size" : args.pop,
              "permutation"     : args.perm,
              "threads"         : args.thread}

    if args.sen == 1:
        config["sen"] = "the dew"
    elif args.sen == 2:
        config["sen"] = "only we"
    
    group_name = args.group
    start_sub  = args.start 
    num_sub    = args.nsub

    if args.win == 'fix':
        DATA_ALL = sio.loadmat("data/subjects_40_sen_fix_win1.0.mat")
    else:
        DATA_ALL = sio.loadmat("data/subjects_40_sen_slide_win1.0_overlap0.5.mat")
    SUBJECT_INFO = pd.read_csv("data/subjects_40_info.csv")

    FEAT_N    = DATA_ALL['FEAT_N']            # Normalized features
    LABEL     = DATA_ALL['LABEL']             # Labels
    LABEL_SEN = DATA_ALL['LABEL_SEN']

    leftout = 1
    testing_acc  = np.zeros(40)
    training_acc = np.zeros(40)
    p_value      = np.zeros(40)

    testing_acc_ga  = np.zeros(40)
    training_acc_ga = np.zeros(40)
    p_value_ga      = np.zeros(40)

    for sub_test in range(start_sub, start_sub + num_sub):

        sub_txt = "R%03d"%(int(SUBJECT_INFO.ID[sub_test]))
        sub_vfi = int(SUBJECT_INFO.VFI[sub_test])
        if sub_vfi > 10:
            sub_group = 'Fatigued'
        else:
            sub_group = 'Healthy'

        if args.wandb:
            run = wandb.init(project  = project_name,
                             group    = group_name,
                             config   = config,
                             name     = sub_txt,
                             tags     = [sub_group, 'GA-SVM', 'Single', 'Sentences'],
                             settings = wandb.Settings(_disable_stats=True, _disable_meta=True),
                             reinit   = True)

        print('\n===No.%d: %s==='%(sub_test+1, sub_txt))
        print('VFI-1:', sub_vfi)

        if args.wandb:
            wandb.log({"subject_info/vfi_1"  : sub_vfi})

        # ===== Load Testing Signals =====
        idx = LABEL_SEN[sub_test][0].flatten() == args.sen
        X_Temp = FEAT_N[sub_test,0][idx,:]
        Y_Temp = LABEL[sub_test,0].flatten()[idx]
        num_signal = np.size(Y_Temp)

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

        # ===== Load Traing Signals =====
        X_TV = np.zeros((0,48))
        Y_TV = np.zeros(0)    
        C_TV = np.zeros(0)
        for sub_train in range(40):
            if sub_train != sub_test:
                idx = LABEL_SEN[sub_train][0].flatten() == args.sen
                x_s = FEAT_N[sub_train,0][idx,:]
                y_s = LABEL[sub_train,0].flatten()[idx]
                n = x_s.shape[0]
                c_s = np.repeat(np.mean(np.array([SUBJECT_INFO.SKINFOLD_SUPRA[sub_train],SUBJECT_INFO.SKINFOLD_INFRA[sub_train]])), n, axis=0)
                X_TV = np.concatenate((X_TV, x_s), axis=0)
                Y_TV = np.concatenate((Y_TV, y_s), axis=0)
                C_TV = np.concatenate((C_TV, c_s), axis=0)

        print('# of Healthy Samples: %d'%(np.sum(Y_TV == -1)))
        print('# of Fatigued Samples: %d'%(np.sum(Y_TV == 1)))

        if args.wandb:
            wandb.log({"exp_info/healthy_samples" : np.sum(Y_TV == -1),
                       "exp_info/fatigued_samples": np.sum(Y_TV ==  1),
                       "exp_info/total_samples"   : np.sum(Y_TV == -1) + np.sum(Y_TV ==  1)})

        # ===== Data loading and preprocessing =====
        # Training and Validation
        # NEED TO REMOVE THE VALIDATION DATA SINCE THEY ARE NOT BEING USED
        X_Train, X_Valid, YC_Train, YC_Valid = train_test_split(X_TV,
                                                                np.transpose([Y_TV, C_TV]),
                                                                test_size=0.1,
                                                                random_state=42)
        Y_Train, C_Train = YC_Train[:,0], YC_Train[:,1]
        Y_Valid, C_Valid = YC_Valid[:,0], YC_Valid[:,1]

        clf = SVC(C=1.0, gamma='scale', kernel='rbf', class_weight='balanced', max_iter=1000, tol=0.001)
        clf.fit(X_Train, Y_Train)
        label_predict = clf.predict(X_Train)

        training_acc[sub_test] = accuracy_score(label_predict, Y_Train)
        print('Training Acc: ', training_acc[sub_test])

        print(Y_Train)
        print(label_predict)
        print(C_Train)

        ret = partial_confound_test(Y_Train, label_predict, C_Train,
                                    cat_y=True, cat_yhat=True, cat_c=False,
                                    cond_dist_method='gam',
                                    progress=False)
        p_value[sub_test] = ret.p
        print('P Value     : ', p_value[sub_test])

        # Evalute rsquared
        df = pd.DataFrame({'x': C_Train, 'y': Y_Train})
        fit = ols('y~C(x)', data=df).fit()
        rsqrd = fit.rsquared.flatten()[0]

        label_predict = clf.predict(X_Test)
        testing_acc[sub_test] = accuracy_score(label_predict, Y_Test)
        print('Testing  Acc: ', testing_acc[sub_test])

        if args.wandb:
            wandb.log({"metrics/train_acc" : training_acc[sub_test],
                       "metrics/test_acc"  : testing_acc[sub_test],
                       "metrics/rsquare"   : rsqrd,
                       "metrics/p_value"   : p_value[sub_test]})

        print('Genetic Algorithm Optimization...')

        if args.wandb:
            num_permu       = wandb.config["permutation"]
            num_generation  = wandb.config["num_generation"]
            population_size = wandb.config["population_size"]
            threads_count   = wandb.config["threads"]
        else:
            num_permu       = config["permutation"]
            num_generation  = config["num_generation"]
            population_size = config["population_size"]
            threads_count   = config["threads"]

        n_threads = threads_count
        pool = ThreadPool(n_threads)
        runner = StarmapParallelization(pool.starmap)

        problem = MyProblem(elementwise_runner=runner)
        problem.load_param(X_Train, Y_Train, C_Train, X_Test, Y_Test, clf, num_permu)

        # Genetic algorithm initialization
        algorithm = NSGA2(pop_size  = population_size,
                          sampling  = FloatRandomSampling(),
                          crossover = SBX(eta=15, prob=0.9),
                          mutation  = PM(eta=20),
                          output    = MultiObjectiveOutput())

        res = minimize(problem,
                       algorithm,
                       ("n_gen", num_generation),
                       callback = MyCallback(args.wandb),
                       verbose=False)

        print('Threads:', res.exec_time)
        pool.close()
        training_acc_ga[sub_test] = res.algorithm.callback.data["train_acc"][-1]
        p_value_ga[sub_test] = res.algorithm.callback.data["p_value"][-1]
        testing_acc_ga[sub_test] = res.algorithm.callback.data["test_acc"][-1]
        rsqrd_best = res.algorithm.callback.data["rsquare"][-1]

        print('Training Acc after GA: ', training_acc_ga[sub_test])
        print('P Value      after GA: ', p_value_ga[sub_test])
        print('Testing  Acc after GA: ', testing_acc_ga[sub_test])

        if args.wandb:
            wandb.log({"metrics/train_acc_ga" : training_acc_ga[sub_test],
                       "metrics/test_acc_ga"  : testing_acc_ga[sub_test],
                       "metrics/p_value_ga"   : p_value_ga[sub_test],
                       "metrics/rsquare_ga"   : rsqrd_best})
            run.finish()