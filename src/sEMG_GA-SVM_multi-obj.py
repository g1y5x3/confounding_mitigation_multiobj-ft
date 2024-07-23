import os
import wandb
import argparse
import numpy as np
from multiprocessing.pool import ThreadPool
from mlconfound.stats import partial_confound_test
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
from pymoo.core.problem import StarmapParallelization, ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.mutation.pm import PM
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.display.multi import MultiObjectiveOutput

from util.sEMGhelpers import load_datafile, LoadTrainTestFeatures

WANDB = os.getenv("WANDB", False)
NAME  = os.getenv("NAME",  "Confounding-Mitigation-In-Deep-Learning")
GROUP = os.getenv("GROUP", "SVM-sEMG")

class MyProblem(ElementwiseProblem):
  def __init__(self, **kwargs):
    super().__init__(n_var=48, n_obj=2, n_constr=0,
                     xl = -2*np.ones(48), xu =  2*np.ones(48),
                     **kwargs)

  def load_data_svm(self, x_train, y_train, c_train, clf, permu):
    # Load informations from the individual classification exerpiment
    # x_train - training features
    # y_train - labels
    # c_train - confounding variables
    # model   - the trained svm model
    self.x_train = x_train
    self.y_train = y_train
    self.c_train = c_train
    self.clf     = clf
    self.permu   = permu

    # dimension of the training feature
    self.n = np.shape(x_train)[0]
    self.d = np.shape(x_train)[1]

  def _evaluate(self, x, out, *args, **kwargs):
    # pymoo initialize the chromosome as a 1-D array which can be converted
    # into matrix for element-wise weight multiplication
    # fw = np.matlib.repmat(x, self.n, 1)
    fw = np.repeat(x.reshape((1,-1)), self.n, axis=0)
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
  def __init__(self) -> None:
    super().__init__()
    self.data["best"] = []

  def notify(self, algorithm):
    print(f"Generation {algorithm.n_gen}")
    self.data["best"].append(algorithm.pop.get("F")[0].min())
    if WANDB: wandb.log({"ga/n_gen"     : algorithm.n_gen,
                         "ga/train_acc" : 1-algorithm.pop.get("F")[0].min(),
                         "ga/p_value"   : 1-algorithm.pop.get("F")[1].min()})

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="sEMG GA-SVM experiments")
  parser.add_argument('-s',      type=int, default=0,   help="start of the subjects")
  parser.add_argument('-nsub',   type=int, default=1,   help="number of subjects to be executed")
  parser.add_argument('-ngen',   type=int, default=1,   help="Number of generation")
  parser.add_argument('-pop',    type=int, default=64,  help='Population size')
  parser.add_argument('-perm',   type=int, default=100, help='Permutation value')
  parser.add_argument('-thread', type=int, default=8,   help='Number of threads')
  args = parser.parse_args()

  #configurations and parameters that doesn't need that are helpful when logged
  config = {"num_generation"  : args.ngen,
            "population_size" : args.pop,
            "permutation"     : args.perm,
            "threads"         : args.thread}

  # X - FEAT_N
  # Y - LABEL
  # C - SUBJECT_SKINFOLD
  FEAT_N, LABEL, SUBJECT_SKINFOLD, VFI_1, SUBJECT_ID = load_datafile("data/subjects_40_v6")

  testing_acc  = np.zeros(40)
  training_acc = np.zeros(40)
  p_value      = np.zeros(40)

  testing_acc_ga  = np.zeros(40)
  training_acc_ga = np.zeros(40)
  p_value_ga      = np.zeros(40)

  start_sub    = args.s
  num_sub      = args.nsub
  for sub_test in range(start_sub, start_sub + num_sub):
    print(SUBJECT_ID[sub_test][0][0])
    sub_txt = "R%03d"%(int(SUBJECT_ID[sub_test][0][0]))
    sub_group = "Fatigued" if int(VFI_1[sub_test][0][0]) > 10 else "Healthy"
    print('\n===No.%d: %s===\n'%(sub_test+1, sub_txt))
    print('VFI-1:', (VFI_1[sub_test][0][0]))


    if WANDB:
      run = wandb.init(project   = NAME,
                        group    = GROUP,
                        config   = config,
                        name     = sub_txt,
                        tags     = [sub_group],
                        settings = wandb.Settings(_disable_stats=True, _disable_meta=True),
                        reinit   = True)
      wandb.log({"subject_info/vfi_1"  : int(VFI_1[sub_test][0][0])})

    X, Y, C, X_Test, Y_Test = LoadTrainTestFeatures(FEAT_N, LABEL, SUBJECT_SKINFOLD, sub_test)

    # Split training and validation (mainly for shuffle validation set technicially ot used here)
    X_Train, X_Valid, YC_Train, YC_Valid = train_test_split(X, np.transpose([Y, C]),
                                                            test_size=0.1,
                                                            random_state=42)
    Y_Train, C_Train = YC_Train[:,0], YC_Train[:,1]

    clf = SVC(C=1.0, gamma='scale', kernel='rbf', class_weight='balanced', max_iter=1000, tol=0.001)
    clf.fit(X_Train, Y_Train)

    # training acc
    label_predict = clf.predict(X_Train)
    training_acc[sub_test] = accuracy_score(label_predict, Y_Train)
    print('Training Acc: ', training_acc[sub_test])

    # p value
    ret = partial_confound_test(Y_Train, label_predict, C_Train,
                                cat_y=True, cat_yhat=True, cat_c=False,
                                cond_dist_method='gam',
                                progress=True)
    p_value[sub_test] = ret.p
    print('P Value     : ', p_value[sub_test])

    # testing acc
    label_predict = clf.predict(X_Test)
    testing_acc[sub_test] = accuracy_score(label_predict, Y_Test)
    print('Testing  Acc: ', testing_acc[sub_test])

    if WANDB: wandb.log({"metrics/train_acc" : training_acc[sub_test],
                         "metrics/test_acc"  : testing_acc[sub_test],
                         "metrics/p_value"   : p_value[sub_test]})

    print('Genetic Algorithm Optimization...')

    num_permu       = config["permutation"]
    num_generation  = config["num_generation"]
    population_size = config["population_size"]
    threads_count   = config["threads"]

    pool = ThreadPool(threads_count)
    runner = StarmapParallelization(pool.starmap)

    problem = MyProblem(elementwise_runner=runner)
    problem.load_data_svm(X_Train, Y_Train, C_Train, clf, num_permu)

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
                   verbose=True)

    print('Threads:', res.exec_time)
    pool.close()

    # Evaluate all the solutions returned by GA
    Xid = np.argsort(res.F[:,0])
    acc_best = 0
    for t in range(np.shape(res.X)[0]):
      w = res.X[Xid[t],:]

      # Evalute training performance
      n = np.shape(X_Train)[0]
      fw = np.repeat(w.reshape((1,-1)), n, axis=0)
      x_train_tf = X_Train * fw
      Y_tf_train = clf.predict(x_train_tf)
      temp_tr_acc = clf.score(x_train_tf, Y_Train)

      # Evaluate the p value from the current predicitons
      ret_ga = partial_confound_test(Y_Train, Y_tf_train, C_Train,
                                     cat_y=True, cat_yhat=True, cat_c=False,
                                     cond_dist_method='gam',
                                     progress=False)
      temp_p_value = ret_ga.p

      # Evaluate the testing performance
      n = np.shape(X_Test)[0]
      fw = np.repeat(w.reshape((1,-1)), n, axis=0)
      x_test_tf = X_Test * fw
      Y_tf_test = clf.predict(x_test_tf)
      temp_te_acc = accuracy_score(Y_tf_test, Y_Test)

      if WANDB: wandb.log({"pareto-front/train_acc": temp_tr_acc,
                           "pareto-front/p_value"  : temp_p_value,
                           "pareto-front/test_acc" : temp_te_acc})

      # Detect if the current chromosome gives the best predictio`n
      if temp_te_acc > acc_best:
        acc_best = temp_te_acc

        training_acc_ga[sub_test] = temp_tr_acc
        p_value_ga[sub_test]      = temp_p_value
        testing_acc_ga[sub_test]  = temp_te_acc

    print("Training Acc after GA: ", training_acc_ga[sub_test])
    print("P Value      after GA: ", p_value_ga[sub_test])
    print("Testing  Acc after GA: ", testing_acc_ga[sub_test])

    if WANDB:
      wandb.log({"metrics/train_acc_cpt" : training_acc_ga[sub_test],
                 "metrics/test_acc_cpt"  : testing_acc_ga[sub_test],
                 "metrics/p_value_cpt"   : p_value_ga[sub_test]})
      run.finish()