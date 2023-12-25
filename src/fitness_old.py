import numpy as np
import wandb
from pymoo.core.callback import Callback
from pymoo.core.problem import ElementwiseProblem
from mlconfound.stats import partial_confound_test


class MyProblem(ElementwiseProblem):

    def __init__(self, **kwargs):
        super().__init__(n_var=48, 
                         n_obj=2,
                         n_constr=0,
                         xl = -2*np.ones(48),
                         xu =  2*np.ones(48),
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
    def __init__(self) -> None:
        super().__init__()
        self.data["best"] = []

    def notify(self, algorithm):
        self.data["best"].append(algorithm.pop.get("F")[0].min())
        wandb.log({"ga/n_gen"     : algorithm.n_gen,
                   "ga/train_acc" : 1-algorithm.pop.get("F")[0].min(),
                   "ga/p_value"   : 1-algorithm.pop.get("F")[1].min()})