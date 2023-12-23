import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import sklearn.metrics as skm
import wandb
import torch

from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tsai.all import *
from fastai.callback.wandb import *
from fastai.layers import *
from sklearn.metrics import accuracy_score
from tqdm.notebook import trange, tqdm

from util.sEMGFeatureLoader import sEMGFeatureDataset

class FeatureMLP(nn.Module):
  def __init__(self):
    super(FeatureMLP, self).__init__()
    self.flatten = Reshape(-1)
    self.mlp = nn.ModuleList()
    self.mlp.append(LinBnDrop(48,50,bn=True, p=0.1, act=get_act_fn(nn.ReLU(inplace=True)), lin_first=False))
    self.mlp.append(LinBnDrop(50,50,bn=True, p=0.2, act=get_act_fn(nn.ReLU(inplace=True)), lin_first=False))
    self.mlp.append(LinBnDrop(50,50,bn=True, p=0.2, act=get_act_fn(nn.ReLU(inplace=True)), lin_first=False))
    self.head = nn.Sequential(LinBnDrop(50,2, bn=False, p=0))
  
  def forward(self, x):
    x = self.flatten(x)
    for mlp in self.mlp: x = mlp(x)
    return self.head(x)
  
def LoadTrainTestFeatures(FEAT, LABEL, sub_test):
  # Load testing samples
  X_Test     = FEAT[sub_test,0]
  Y_Test     = LABEL[sub_test,0].flatten()
  print(f'# of Testing Samples {len(Y_Test)}')

  # Load training samples
  X_Train = np.zeros((0,48))
  Y_Train = np.zeros(0)    
  for sub_train in range(40):
    if sub_train != sub_test:
      x_s = FEAT[sub_train,0]
      y_s = LABEL[sub_train,0].flatten()
      X_Train = np.concatenate((X_Train, x_s), axis=0)
      Y_Train = np.concatenate((Y_Train, y_s), axis=0)

  print('# of Healthy Samples: %d'%(np.sum(Y_Train == -1)))
  print('# of Fatigued Samples: %d'%(np.sum(Y_Train == 1)))   
  
  return X_Train, Y_Train, X_Test, Y_Test

# mainly just for the sake of not keeping the copy of DATA_ALL
def load_datafile(file):
  DATA_ALL = sio.loadmat(file)
  FEAT_N           = DATA_ALL['FEAT_N']            # Normalized features
  LABEL            = DATA_ALL['LABEL']             # Labels
  VOWEL            = DATA_ALL['LABEL_VOWEL']       # Type of Vowels
  VFI_1            = DATA_ALL['SUBJECT_VFI']       # VFI-1 Score
  SUBJECT_ID       = DATA_ALL['SUBJECT_ID']        # Sujbect ID
  SUBJECT_SKINFOLD = DATA_ALL['SUBJECT_SKINFOLD']  # Subject Skinfold Thickness
  return FEAT_N, LABEL, SUBJECT_SKINFOLD, VOWEL, VFI_1, SUBJECT_ID

# environment variable for the experiment
WANDB = os.getenv("WANDB", False)
GROUP = os.getenv("GROUP", "tests")

if __name__ == "__main__":
  # X - FEAT_N
  # Y - LABEL
  # C - SUBJECT_SKINFOLD
  FEAT_N, LABEL, _, VFI_1, SUBJECT_ID = load_datafile("data/subjects_40_v6")

  for sub_test in range(10):
    sub_txt = "R%03d"%(int(SUBJECT_ID[sub_test][0][0]))
    print('Test Subject %s:'%(sub_txt))
    print('VFI-1:', (VFI_1[sub_test][0][0]))
    # if int(VFI_1[sub_test][0][0]) > 10:
    #   sub_group = 'Fatigued'
    # else:
    #   sub_group = 'Healthy'

    # if WANDB:
    #     run = wandb.init(project="sEMG Leave-One-Out Classification 40 100 Feature Base TSAI",
    #                      group=sub_group,
    #                      name=sub_txt
    #                      reinit=True,
    #                      )

    # #  Load training/testing features
    # X_Train, Y_Train, X_Test, Y_Test = LoadTrainTestFeatures(FEAT_N, LABEL, sub_test)

    # #  Data initialization for fastai/pytorch
    # splits = get_splits(Y_Train, valid_size=.1, stratify=True, random_state=23, shuffle=True, show_plot=False)
    # tfms  = [None, [Categorize()]]
    # dsets = TSDatasets(X_Train, Y_Train, tfms=tfms, splits=splits)

    # dls = TSDataLoaders.from_dsets(dsets.train,
    #                                dsets.valid,
    #                                shuffle_train=True,
    #                                bs=32,
    #                                num_workers=0)

    # # dls.show_batch()

    # #  Model definitions
    # model = FeatureMLP()

    # if wand_config == 1:
    #     learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=accuracy, cbs=WandbCallback())
    # else: