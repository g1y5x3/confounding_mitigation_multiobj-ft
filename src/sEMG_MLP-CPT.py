import os
import wandb
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
from fastai.learner import Learner
from fastai.data.core import DataLoaders
from fastai.callback.wandb import WandbCallback
from tsai.all import get_splits, MLP
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score
# from mlconfound.stats import partial_confound_test

from util.sEMGhelpers import load_datafile, partition
from cpt import conditional_log_likelihood, generate_X_CPT_MC, cpt_p_pearson

# environment variable for the experiment
WANDB = os.getenv("WANDB", False)
NAME  = os.getenv("NAME",  "Confounding-Mitigation-In-Deep-Learning")
GROUP = os.getenv("GROUP", "MLP-sEMG-CPT")

class sEMGDataset(Dataset):
  def __init__(self, X_Train, Y_Train, C_Train=None, index=None):
    self.X_Train = X_Train
    self.Y_Train = Y_Train
    self.C_Train = C_Train
    self.index   = index

  def __len__(self):
    return len(self.Y_Train)

  def __getitem__(self, idx):
    x   = torch.tensor(self.X_Train[idx,:], dtype=torch.float32)
    c   = torch.tensor(self.C_Train[idx],   dtype=torch.float32)
    i   = torch.tensor(self.index[idx],     dtype=torch.int)
    y   = torch.tensor(self.Y_Train[idx],   dtype=torch.long)
    return (x, c, i), y

# model is the same but passing an additional input variable
class MLPC(MLP):
  def __init__(self, c_in, c_out, seq_len, layers,
               ps=[0.1, 0.2, 0.2], act=nn.ReLU(inplace=True), use_bn=False, bn_final=False, lin_first=False, fc_dropout=0., y_range=None):
    super().__init__(c_in, c_out, seq_len, layers, ps, act, use_bn, bn_final, lin_first, fc_dropout, y_range)

  def forward(self, x_c_idx):
    x, c, idx = x_c_idx
    return (super().forward(x), c, idx)

class CrossEntropyLoss(nn.Module):
  def __init__(self):
    super().__init__()

  def __call__(self, yhat_c_idx, y):
    yhat, _, _ = yhat_c_idx
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
  print(bs)
  cond_log_like_mat = conditional_log_likelihood(X=c.numpy(), C=y.numpy(), xdtype='categorical')
  print(cond_log_like_mat.shape)
  Pi_init = generate_X_CPT_MC(mcmc_steps*5, cond_log_like_mat, np.arange(bs, dtype=int), random_state)
  print(Pi_init.shape)

  def workhorse(c, _random_state):
    # batched os job_batch for efficient parallelization
    Pi = generate_X_CPT_MC(mcmc_steps, cond_log_like_mat, Pi_init, random_state=_random_state)
    return c[Pi]
  rng = np.random.default_rng(random_state)
  random_states = rng.integers(np.iinfo(np.int32).max, size=num_perm)
  print(Pi_init.shape)
  c_pi_np = np.array(Parallel(n_jobs=-1)(delayed(workhorse)(c, i) for i in random_states))
  c_pi = torch.tensor(c_pi_np, dtype=torch.float32)
  # compute p-value
  t_yhat_c = distance_correlation(yhat.reshape([bs, -1]), c.reshape([bs, -1])).repeat(num_perm)
  t_yhat_cpi = torch.zeros(num_perm)
  for i in range(num_perm):
    t_yhat_cpi[i] = distance_correlation(yhat.reshape([bs, -1]), c_pi[i,:].reshape([bs, -1]))

  return torch.sigmoid(t_yhat_cpi - t_yhat_c).mean()

# TODO how to make it faster?
class CrossEntropyCPTLoss(nn.Module):
  def __init__(self, cond_like_mat, mcmc_steps=50, random_state=123, num_perm=1000):
    super().__init__()
    self.mcmc_steps        = mcmc_steps   # this is the default value used inside original CPT function
    self.random_state      = random_state
    self.num_perm          = num_perm
    self.cond_log_like_mat = cond_like_mat

  def __call__(self, yhat_c_idx, y):
    yhat, c, idx = yhat_c_idx
    return F.cross_entropy(yhat, y)

def accuracy(preds_confound_index, targets):
  preds, _, _ = preds_confound_index
  return (preds.argmax(dim=-1) == targets).float().mean()

def pvalue_dcor(preds_confound, targets):
  preds, confound = preds_confound
  return cpt_p_dcor(confound, preds, targets)

def pvalue_pearson(preds_confound, targets):
  preds, confound = preds_confound
  p, _ = cpt_p_pearson(confound.numpy(), preds.argmax(dim=1).numpy(), targets.numpy(), random_state=123, num_perm=1000, dtype='categorical')
  return p

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="sEMG MLP CPT experiments")
  parser.add_argument('-bs', type=int, default=256, help="batch size")
  parser.add_argument('-nw', type=int, default=2,   help="number of workers")
  args = parser.parse_args()

  FEAT_N, LABEL, SUBJECT_SKINFOLD, VFI_1, SUBJECT_ID = load_datafile("data/subjects_40_v6")

  train_acc = np.zeros(40)
  valid_acc = np.zeros(40)
  test_acc  = np.zeros(40)
  p_value   = np.zeros(40)
  for sub_test in range(1, 40):
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
    Y_Train = np.where(Y_Train == -1, 0, 1)
    Y_Test  = np.where(Y_Test  == -1, 0, 1)

    # Setting "stratify" to True ensures that the relative class frequencies are approximately preserved in each train and validation fold.
    cond_like_mat = conditional_log_likelihood(X=C_Train, C=Y_Train, xdtype='categorical')
    splits = get_splits(Y_Train, valid_size=.1, stratify=True, random_state=123, shuffle=True, show_plot=False)
    dsets_train = sEMGDataset(X_Train[splits[0],:], Y_Train[splits[0]], C_Train[splits[0]], splits[0])
    dsets_valid = sEMGDataset(X_Train[splits[1],:], Y_Train[splits[1]], C_Train[splits[1]], splits[1])
    dsets_test  = sEMGDataset(X_Test, Y_Test, C_Test, list(np.arange(len(X_Test))))

    # NOTE disable shuffling to utilize the conditional likelihood matrix estimated upfront
    print(f"batch size: {args.bs}")
    print(f"# of workers: {args.nw}")
    dls = DataLoaders.from_dsets(dsets_train, dsets_valid, shuffle=False, bs=args.bs, num_workers=args.nw, pin_memory=True)

    # This model is pre-defined in https://timeseriesai.github.io/tsai/models.mlp.html
    model = MLPC(c_in=1, c_out=2, seq_len=48, layers=[50, 50, 50], use_bn=True)
    learn = Learner(dls,
                    model,
                    loss_func=CrossEntropyCPTLoss(cond_like_mat),
                    metrics=[accuracy],
                    cbs=cbs)

    learn.lr_find()
    learn.fit_one_cycle(50, lr_max=1e-3)

    # Training accuracy
    train_output, train_targets = learn.get_preds(dl=dls.train, with_loss=False)
    train_preds, train_c, _ = train_output
    train_acc[sub_test] = accuracy_score(train_targets, train_preds.argmax(dim=1))
    print(f"Training acc   : {train_acc[sub_test]}")

    # P-value (only makes sense to report for training)
    p, _ = cpt_p_pearson(train_c.numpy(), train_preds.argmax(dim=1).numpy(), train_targets.numpy(), cond_like_mat[splits[0],:][:,splits[0]],
                                mcmc_steps=100, random_state=None, num_perm=2000, dtype='categorical')
    p_value[sub_test] = p
    print(f"P Value        : {p}")

    # Validation accuracy
    valid_output, valid_targets = learn.get_preds(dl=dls.valid, with_loss=False)
    valid_preds, valid_c, _ = valid_output
    valid_acc[sub_test] = accuracy_score(valid_targets, valid_preds.argmax(dim=1))
    print(f"Validation acc : {valid_acc[sub_test]}")

    # Testing accuracy
    dls_test = dls.new(dsets_test)
    learn.loss_func = CrossEntropyLoss()
    learn.metrics = accuracy
    test_output, test_targets = learn.get_preds(dl=dls_test, with_loss=False)
    test_preds, test_c, _ = test_output
    test_acc[sub_test] = accuracy_score(test_targets, test_preds.argmax(dim=1))
    print(f"Testing acc : {test_acc[sub_test]}")

    if WANDB: wandb.log({"subject_info/vfi_1" : int(VFI_1[sub_test][0][0]),
                         "metrics/train_acc_opt"  : train_acc[sub_test],
                         "metrics/test_acc_opt"   : test_acc[sub_test],
                         "metrics/p_value_opt"    : p_value[sub_test]})