import os
import wandb
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
from fastai.learner import Learner
from fastai.data.core import DataLoaders
from fastai.callback.wandb import WandbCallback
from tsai.all import get_splits, MLP
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score
from mlconfound.stats import partial_confound_test
#
from util.sEMGhelpers import load_datafile, partition
from cpt import conditional_log_likelihood, generate_X_CPT_MC, cpt_p_pearson

class sEMGDataset(Dataset):
  def __init__(self, X_Train, Y_Train, C_Train=None):
    self.X_Train = X_Train
    self.Y_Train = Y_Train
    self.C_Train = C_Train

  def __len__(self):
    return len(self.Y_Train)

  def __getitem__(self, idx):
    x = torch.tensor(self.X_Train[idx,:], dtype=torch.float32)
    y = torch.tensor(self.Y_Train[idx],   dtype=torch.long)
    c = torch.tensor(self.C_Train[idx],   dtype=torch.float32)
    return (x, c), y

# model is the same but passing an additional input variable
class MLPC(MLP):
  def __init__(self, c_in, c_out, seq_len, layers,
               ps=[0.1, 0.2, 0.2], act=nn.ReLU(inplace=True), use_bn=False, bn_final=False, lin_first=False, fc_dropout=0., y_range=None):
    super().__init__(c_in, c_out, seq_len, layers, ps, act, use_bn, bn_final, lin_first, fc_dropout, y_range)

  def forward(self, x_c):
    x, c = x_c
    return (super().forward(x), c)

class CrossEntropyLoss(nn.Module):
  def __init__(self):
    super().__init__()

  def __call__(self, yhat_c, y):
    print(yhat_c)
    print(y)
    yhat, _ = yhat_c
    return F.cross_entropy(yhat, y)

# TODO vectorize this implementation
# https://github.com/zhenxingjian/Partial_Distance_Correlation/blob/b088801996acefe38a67dff59bb8cbe3b20c7d91/Partial_Distance_Correlation.ipynb
def distance_correlation(c, y):
  matrix_a = torch.sqrt(torch.sum(torch.square(c.unsqueeze(0) - c.unsqueeze(1)), dim = -1) + 1e-12)
  matrix_b = torch.sqrt(torch.sum(torch.square(y.unsqueeze(0) - y.unsqueeze(1)), dim = -1) + 1e-12)

  matrix_A = matrix_a - torch.mean(matrix_a, dim = 0, keepdims= True) - torch.mean(matrix_a, dim = 1, keepdims= True) + torch.mean(matrix_a)
  matrix_B = matrix_b - torch.mean(matrix_b, dim = 0, keepdims= True) - torch.mean(matrix_b, dim = 1, keepdims= True) + torch.mean(matrix_b)

  gamma_XY = torch.sum(matrix_A * matrix_B)/ (matrix_A.shape[0] * matrix_A.shape[1])
  gamma_XX = torch.sum(matrix_A * matrix_A)/ (matrix_A.shape[0] * matrix_A.shape[1])
  gamma_YY = torch.sum(matrix_B * matrix_B)/ (matrix_A.shape[0] * matrix_A.shape[1])

  correlation_r = gamma_XY/torch.sqrt(gamma_XX * gamma_YY + 1e-9)
  return correlation_r

def cpt_p_dcor(c, yhat, y, mcmc_steps=50, random_state=123, num_perm=1000):
  # sampling permutations of c
  bs = y.shape[0]
  cond_log_like_mat = conditional_log_likelihood(X=c.numpy(), C=y.numpy(), xdtype='categorical')
  Pi_init = generate_X_CPT_MC(mcmc_steps*5, cond_log_like_mat, np.arange(bs, dtype=int), random_state)

  def workhorse(c, _random_state):
    # batched os job_batch for efficient parallelization
    Pi = generate_X_CPT_MC(mcmc_steps, cond_log_like_mat, Pi_init, random_state=_random_state)
    return c[Pi]
  rng = np.random.default_rng(random_state)
  random_states = rng.integers(np.iinfo(np.int32).max, size=num_perm)
  c_pi = torch.tensor(np.array(Parallel(n_jobs=-1)(delayed(workhorse)(c, i) for i in random_states)), dtype=torch.float32)
  # compute p-value
  t_yhat_c = distance_correlation(yhat.reshape([bs, -1]), c.reshape([bs, -1])).repeat(num_perm)
  t_yhat_cpi = torch.zeros(num_perm)
  for i in range(num_perm):
    t_yhat_cpi[i] = distance_correlation(yhat.reshape([bs, -1]), c_pi[i,:].reshape([bs, -1]))

  return torch.sigmoid(t_yhat_cpi - t_yhat_c).mean()

# TODO how to make it faster?
class CrossEntropyCPTLoss(nn.Module):
  def __init__(self, mcmc_steps=50, random_state=123, num_perm=1000):
    super().__init__()
    self.mcmc_steps = mcmc_steps   # this is the default value used inside original CPT function
    self.random_state = random_state
    self.num_perm = num_perm

  def __call__(self, yhat_c, y):
    yhat, c = yhat_c
    p = cpt_p_dcor(c, yhat, y, self.mcmc_steps, self.random_state, self.num_perm)
    return F.cross_entropy(yhat, y) - p
    # return F.cross_entropy(yhat, y)

def accuracy(preds_confound, targets):
  preds, _, = preds_confound
  return (preds.argmax(dim=-1) == targets).float().mean()

def pvalue_dcor(preds_confound, targets):
  preds, confound = preds_confound
  return cpt_p_dcor(confound, preds, targets)

def pvalue_pearson(preds_confound, targets):
  preds, confound = preds_confound
  # print(preds.argmax(dim=1), confound, targets)
  p, _ = cpt_p_pearson(confound.numpy(), preds.argmax(dim=1).numpy(), targets.numpy(), random_state=123, num_perm=1000, dtype='categorical')
  # print(p)
  return p

# environment variable for the experiment
WANDB = os.getenv("WANDB", False)
NAME  = os.getenv("NAME",  "Confounding-Mitigation-In-Deep-Learning")
GROUP = os.getenv("GROUP", "MLP-sEMG-CPT")

# NOTE
# Most of the deep learning pipeline here are implemented through fastai API, tsai is just another
# API wrapper that provides utilities for time-domain inputs
if __name__ == "__main__":
  # X - FEAT_N
  # Y - LABEL
  # C - SUBJECT_SKINFOLD
  FEAT_N, LABEL, SUBJECT_SKINFOLD, VFI_1, SUBJECT_ID = load_datafile("data/subjects_40_v6")

  # NOTE
  # For the neural networks implementation, a high-level API was used in order to minimize implementation
  # tsai is wrapped around fastai's API but it has a better numpy interface
  # more reference can be found in https://timeseriesai.github.io/tsai/
  train_acc = np.zeros(40)
  test_acc  = np.zeros(40)
  p_value   = np.zeros(40)
  for sub_test in range(0, 1):
    sub_txt = "R%03d"%(int(SUBJECT_ID[sub_test][0][0]))
    sub_group = "Fatigued" if int(VFI_1[sub_test][0][0][0]) > 10 else "Healthy"
    print('\n===No.%d: %s===\n'%(sub_test+1, sub_txt))
    print('VFI-1:', (VFI_1[sub_test][0][0]))

    cbs = None
    if WANDB:
      run = wandb.init(project=NAME, group=GROUP, name=sub_txt, tags=[sub_group], reinit=True)
      cbs = WandbCallback(log_preds=False)

    print("Loading training and testing set")
    X_Train, Y_Train, C_Train, X_Test, Y_Test, C_Test = partition(FEAT_N, LABEL, SUBJECT_SKINFOLD, sub_test)
    # convert labels from [-1, 1] to [0, 1] so the probability density function estimation will be consistent with the dataset transformation
    Y_Train = np.where(Y_Train == -1, 0, 1)
    Y_Test  = np.where(Y_Test == -1, 0, 1)

    # Setting "stratify" to True ensures that the relative class frequencies are approximately preserved in each train and validation fold.
    splits = get_splits(Y_Train, valid_size=.1, stratify=True, random_state=123, shuffle=True, show_plot=False)

    dsets_train = sEMGDataset(X_Train[splits[0],:], Y_Train[splits[0]], C_Train[splits[0]])
    dsets_valid = sEMGDataset(X_Train[splits[1],:], Y_Train[splits[1]], C_Train[splits[1]])
    dsets_test  = sEMGDataset(X_Test, Y_Test, C_Test)

    dls = DataLoaders.from_dsets(dsets_train, dsets_valid, shuffle=True, bs=128, num_workers=2, pin_memory=True)

    # This model is pre-defined in https://timeseriesai.github.io/tsai/models.mlp.html
    model = MLPC(c_in=1, c_out=2, seq_len=48, layers=[50, 50, 50], use_bn=True)
    print(model)

    learn = Learner(dls, model, loss_func=CrossEntropyCPTLoss(), metrics=[accuracy, pvalue_dcor, pvalue_pearson], cbs=cbs)

    # Basically apply some tricks to make it converge faster
    # https://docs.fast.ai/callback.schedule.html#learner.lr_find
    # https://docs.fast.ai/callback.schedule.html#learner.fit_one_cycle
    learn.lr_find()
    learn.fit_one_cycle(1, lr_max=1e-3)

    # Training accuracy
    train_output, train_targets = learn.get_preds(dl=dls.train, with_loss=False)
    train_preds, train_c = train_output
    train_acc[sub_test] = accuracy_score(train_targets, train_preds.argmax(dim=1))
    print(f"Training acc: {train_acc[sub_test]}")

    # P-value
    ret = partial_confound_test(train_targets.numpy(), train_preds.argmax(dim=1).numpy(), train_c.numpy(),
                                cat_y=True, cat_yhat=True, cat_c=False,
                                cond_dist_method='gam',
                                progress=True)
    p_value[sub_test] = ret.p
    print(f"P Value     : {p_value[sub_test]}")

    # this is extremely slow, even on the gpu dute to computing the pairwise distance over a 6000+ training data size
    # print(f"P Value (dCor): {cpt_p_dcor(c=train_c, yhat=train_preds, y=train_targets)}")

    # Testing accuracy
    dls_test = dls.new(dsets_test)
    # the loss function is still being called internally during inference probably for some internal tracking in the API,
    # hence here switch to a different loss function since there's no point calculating the cpt p value
    learn.loss_func = CrossEntropyLoss()
    learn.metrics = accuracy
    test_output, test_targets = learn.get_preds(dl=dls_test, with_loss=False)
    test_preds, test_c = test_output
    test_acc[sub_test] = accuracy_score(test_targets, test_preds.argmax(dim=1))
    print(f"Testing acc : {test_acc[sub_test]}")

    if WANDB: wandb.log({"subject_info/vfi_1" : int(VFI_1[sub_test][0][0]),
                         "metrics/train_acc_opt"  : train_acc[sub_test],
                         "metrics/test_acc_opt"   : test_acc[sub_test],
                         "metrics/p_value_opt"    : p_value[sub_test]})