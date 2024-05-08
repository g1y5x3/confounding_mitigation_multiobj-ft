import torch
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset

class sEMGFeatureDataset(Dataset):
  def __init__(self, Feature,Label, transform=None, verbose=False):
    self.transform = transform
    self.feature = Feature
    self.label   = Label
  
  def __len__(self):
      return np.shape(self.label)[0]
  
  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    
    if self.label[idx] == 1:
      label = np.array([0, 1])
    elif self.label[idx] == -1:
      label = np.array([1, 0])
    
    feature = torch.tensor(self.feature[idx,:], dtype=torch.float32)
    label   = torch.tensor(label, dtype=torch.float32)
    sample = {'feature': feature, "label": label}
    
    return sample

class sEMGSignalDataset(Dataset):
  def __init__(self, signals, labels):
    self.signals = signals
    self.labels = labels

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    signal = torch.tensor(self.signals[idx,:,:], dtype=torch.float32)
    label = torch.tensor(self.labels[idx,:], dtype=torch.float32)
    return signal, label

if __name__ == "__main__":
  data_all = sio.loadmat("data/subjects_40_v6.mat")
  print(data_all)