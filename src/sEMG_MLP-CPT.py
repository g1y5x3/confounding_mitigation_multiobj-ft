import os
import wandb
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
from tsai.all import get_splits, Categorize, TSDatasets, MLP
from fastai.data.core import DataLoaders
from fastai.learner import Learner
from fastai.metrics import accuracy
from fastai.callback.wandb import WandbCallback
from sklearn.metrics import accuracy_score
from util.sEMGhelpers import load_datafile, partition
from mlconfound.stats import partial_confound_test
from cpt import conditional_log_likelihood, generate_X_CPT_MC

# The easiest way to integrate with fastai's API is to redefine the class for
# dataloader, model, loss
class sEMGDataset(Dataset):
  def __init__(self, X_Train, Y_Train, C_Train):
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

# model is exactly the same other than passing through an additional input variable
class MLPC(MLP):
  def __init__(self, c_in, c_out, seq_len, layers=..., ps=[0.1, 0.2, 0.2], act=nn.ReLU(inplace=True), use_bn=False, bn_final=False, lin_first=False, fc_dropout=0., y_range=None):
    super().__init__(c_in, c_out, seq_len, layers, ps, act, use_bn, bn_final, lin_first, fc_dropout, y_range)

  def forward(self, xc):
    x, c = xc
    return (super().forward(x), c)

class CrossEntropyCPTLoss(nn.Module):
  # When the loss object is being initiated, it estimates the prob density function of q(C|Y)
  # y - training labels
  # c - confounding variables
  def __init__(self, y, c):
    super().__init__()
    print("Estimating Q(C|Y)...")
    print(y.shape)
    print(y)
    print(c.shape)
    print(c)
    # estimate the condtional log likelihood function

  def forward(self, preds_confound, targets):
    preds, confound = preds_confound
    # print(preds)
    # print(targets)
    return F.cross_entropy(preds, targets)

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
  for sub_test in range(20, 21):
    sub_txt = "R%03d"%(int(SUBJECT_ID[sub_test][0][0]))
    sub_group = "Fatigued" if int(VFI_1[sub_test][0][0][0]) > 10 else "Healthy"
    print('\n===No.%d: %s===\n'%(sub_test+1, sub_txt))
    print('VFI-1:', (VFI_1[sub_test][0][0]))

    cbs = None
    if WANDB:
      run = wandb.init(project=NAME, group=GROUP, name=sub_txt, tags=[sub_group], reinit=True)
      cbs = WandbCallback(log_preds=False)

    print("Loading training and testing set")
    X_Train, Y_Train, C_Train, X_Test, Y_Test = partition(FEAT_N, LABEL, SUBJECT_SKINFOLD, sub_test)
    # convert labels from [-1, 1] to [0, 1] so the probability density function estimation will be consistent with the dataset transformation
    Y_Train = np.where(Y_Train == -1, 0, 1)

    # Setting "stratify" to True ensures that the relative class frequencies are approximately preserved in each train and validation fold.
    splits = get_splits(Y_Train, valid_size=.1, stratify=True, random_state=123, shuffle=True, show_plot=False)
    tfms   = [None, [Categorize()]]
    # dsets       = TSDatasets(X_Train, Y_Train, tfms=tfms, splits=splits)
    dsets_train = sEMGDataset(X_Train[splits[0],:], Y_Train[splits[0]], C_Train[splits[0]])
    dsets_valid = sEMGDataset(X_Train[splits[1],:], Y_Train[splits[1]], C_Train[splits[1]])

    # dsets_train = TSDatasets(X_Train, Y_Train, tfms=tfms) # keep an unsplit copy for computing the p-value
    dsets_test  = TSDatasets(X_Test,  Y_Test,  tfms=tfms)
    dls       = DataLoaders.from_dsets(dsets_train, dsets_valid, shuffle_train=True, bs=32, num_workers=2)
    # dls_train = dls.new(dsets_train)
    dls_test  = dls.new(dsets_test)

    # This model is pre-defined in https://timeseriesai.github.io/tsai/models.mlp.html
    model = MLPC(c_in=1, c_out=2, seq_len=48, layers=[50, 50, 50], use_bn=True)
    print(model)

    learn = Learner(dls, model, loss_func=CrossEntropyCPTLoss(y=Y_Train, c=C_Train), metrics=None, cbs=cbs)

    # Basically apply some tricks to make it converge faster
    # https://docs.fast.ai/callback.schedule.html#learner.lr_find
    # https://docs.fast.ai/callback.schedule.html#learner.fit_one_cycle
    learn.lr_find()
    learn.fit_one_cycle(50, lr_max=1e-3)

    # train_preds, train_targets = learn.get_preds(dl=dls_train)
    # train_acc[sub_test] = accuracy_score(train_targets, train_preds.argmax(dim=1))
    print(f"Training acc: {train_acc[sub_test]}")

    # ret = partial_confound_test(train_targets.numpy(), train_preds.argmax(dim=1).numpy(), C_Train,
    #                             cat_y=True, cat_yhat=True, cat_c=False,
    #                             cond_dist_method='gam',
    #                             progress=True)
    # p_value[sub_test] = ret.p
    # print(f"P Value     : {p_value[sub_test]}")

    test_preds, test_targets = learn.get_preds(dl=dls_test)
    test_acc[sub_test] = accuracy_score(test_targets, test_preds.argmax(dim=1))
    print(f"Testing acc : {test_acc[sub_test]}")

    if WANDB: wandb.log({"subject_info/vfi_1" : int(VFI_1[sub_test][0][0]),
                         "metrics/train_acc_opt"  : train_acc[sub_test],
                         "metrics/test_acc_opt"   : test_acc[sub_test],
                         "metrics/p_value_opt"    : p_value[sub_test]})