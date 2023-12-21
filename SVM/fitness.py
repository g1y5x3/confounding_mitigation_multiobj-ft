import os
import wandb
import numpy as np
from pymoo.core.callback import Callback
from pymoo.core.problem import ElementwiseProblem
from mlconfound.stats import partial_confound_test

WANDB = os.getenv("WANDB", False)

class MyProblem(ElementwiseProblem):

  def __init__(self, **kwargs):
    super().__init__(n_var=48, 
                     n_obj=2,
                     n_constr=0,
                     xl = -2*np.ones(48), # here weights are bounded between -2 to 2
                     xu =  2*np.ones(48),
                     **kwargs)

  # The optimization is purely based on training sets and p-value from the training
  # predictions.
  # The testing set is passed just for the class object to store it
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
    fw = np.repeat(x.reshape((1,-1)), self.n, axis=0)
    x_train_tf = self.x_train * fw

    # first objective - SVM training accuracy
    f1 = 1 - self.clf.score(x_train_tf, self.y_train)

    y_hat = self.clf.predict(x_train_tf)
    ret = partial_confound_test(self.y_train, y_hat, self.c_train,
                                cat_y=True, cat_yhat=True, cat_c=False,
                                cond_dist_method='gam', 
                                num_perms=self.permu, mcmc_steps=50,
                                n_jobs=-1,
                                progress=False)
    # second objective - 1 P-Value from CPT  
    f2 = 1 - ret.p 

    out['F'] = [f1, f2]

# model evaluations at the end of each generation
# after optimization, we receive a parento front - a set of solutions
# therefore we evaluate each solution and report the ones the gives
# the best testing accuracy
class MyCallback(Callback):
  def __init__(self) -> None:
    super().__init__()
    self.data["train_acc"] = []
    self.data["test_acc"] = []
    self.data["p_value"] = []
    self.data["predict"] = []

  def notify(self, algorithm):
    F = algorithm.pop.get("F")
    X = algorithm.pop.get("X")
    
    # Evaluate the results from GA
    Xid = np.argsort(F[:,0])
    acc_best = 0        
    tr_acc_best  = 0 
    p_value_best = 0 
    te_acc_best  = 0 
    predict_best = []
    for t in range(np.shape(X)[0]):

      w = X[Xid[t],:]
      fw = np.repeat(w.reshape((1,-1)), algorithm.problem.n, axis=0)

      # Evalute the training performance
      x_train_tf  = algorithm.problem.x_train * fw
      y_train_tf  = algorithm.problem.clf.predict(x_train_tf)
      temp_tr_acc = algorithm.problem.clf.score(x_train_tf, 
                                                algorithm.problem.y_train)

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
      fw = np.repeat(w.reshape((1,-1)), n, axis=0)
      x_test_tf = algorithm.problem.x_test * fw
      temp_te_acc = algorithm.problem.clf.score(x_test_tf, 
                                                algorithm.problem.y_test)

      # TODO upload parento front

      if temp_te_acc > acc_best:
        acc_best = temp_te_acc 
        tr_acc_best  = temp_tr_acc 
        p_value_best = temp_p_value
        te_acc_best  = temp_te_acc
        predict_best = algorithm.problem.clf.predict(x_test_tf)

    # Give an update at the end of generation
    print(f"n_gen:     {algorithm.n_gen}")
    print(f"train_acc: {tr_acc_best}")
    print(f"p_value:   {p_value_best}")
    print(f"test_acc:  {te_acc_best}")
    if WANDB:
      wandb.log({"GA/n_gen"     : algorithm.n_gen,
                 "GA/train_acc" : tr_acc_best,
                 "GA/p_value"   : p_value_best,
                 "GA/test_acc"  : te_acc_best})

    self.data["train_acc"].append(tr_acc_best)
    self.data["p_value"].append(p_value_best)
    self.data["test_acc"].append(te_acc_best)
    self.data["predict"].append(predict_best)
