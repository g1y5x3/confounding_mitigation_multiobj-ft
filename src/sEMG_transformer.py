import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from util.sEMGhelpers import load_raw_signals
from util.sEMGFeatureLoader import sEMGSignalDataset

class simpleEMGtransformer(nn.Module):
  def __init__(self, d_model=64, nhead=4, dim_feedforward=512):
    super().__init__()
    # maybe have to reduce the seq using conv1d
    self.input_project = nn.Linear(4, d_model)
    self.encoder = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, norm_first=True)
    self.pool = nn.AdaptiveAvgPool1d(1)
    self.fc = nn.Linear(d_model, 2)

  def forward(self, x):
    x = self.input_project(x)
    x = self.encoder(x)
    x = x.permute(0,2,1)
    x = self.pool(x)
    x = x.flatten(start_dim=1)
    output = self.fc(x)
    return output

if __name__ == "__main__":

  # signal pre-processing
  signals, labels, VFI1, sub_id, sub_skinfold = load_raw_signals("data/subjects_40_v6.mat")

  X, Y = [], []
  for i in range(40):
    # normalize the signal subject-wise
    x = np.stack(signals[i], axis=2)
    x_means = np.mean(x, axis=(0,1))
    x_stds = np.std(x, axis=(0,1))
    x_norm = (x - x_means[np.newaxis, np.newaxis, :]) / x_stds[np.newaxis, np.newaxis, :]

    # one-hot encode the binary labels
    N = labels[i][0].shape[0]
    mapped_indices = (labels[i][0] == 1).astype(int)
    y_onehot = np.zeros((N, 2))
    y_onehot[np.arange(N), mapped_indices.flatten()] = 1

    X.append(x_norm)
    Y.append(y_onehot)

  X, Y = np.concatenate(X, axis=0), np.concatenate(Y, axis=0)
  print(f"X {X.shape}")
  print(f"Y {Y.shape}")

  # shuffle indices
  num_samples = X.shape[0]
  indices = np.arange(num_samples)
  np.random.shuffle(indices)

  # split indices for train and test
  split_idx = int(num_samples*0.9)
  train_idx, valid_idx = indices[:split_idx], indices[split_idx:]

  X_train, X_valid = X[train_idx], X[valid_idx]
  Y_train, Y_valid = Y[train_idx], Y[valid_idx]
  print(f"X_train {X_train.shape}")
  print(f"Y_train {Y_train.shape}")
  print(f"X_valid {X_valid.shape}")
  print(f"Y_valid {Y_valid.shape}")

  bsz = 32

  dataset_train = sEMGSignalDataset(X_train, Y_train)
  dataset_valid = sEMGSignalDataset(X_valid, Y_valid)
  dataloader_train = DataLoader(dataset_train, batch_size=bsz, shuffle=True)
  dataloader_valid = DataLoader(dataset_valid, batch_size=bsz, shuffle=False)

  model = simpleEMGtransformer()
  model.to("cuda")

  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.AdamW(model.parameters())

  writer = SummaryWriter()
  for epoch in tqdm(range(250), desc="Training Epochs"):
    loss_train = 0
    model.train()
    for batch, (inputs, targets) in enumerate(dataloader_train):
      optimizer.zero_grad()

      inputs, targets = inputs.to("cuda"), targets.to("cuda")
      outputs = model(inputs)
      loss = criterion(outputs, targets)

      loss.backward()
      optimizer.step()

      loss_train += loss.item()

    writer.add_scalar("loss/train", loss_train/len(dataset_train), epoch)

    model.eval()
    loss_valid = 0
    correct = 0
    for inputs, targets in dataloader_valid:
      inputs, targets = inputs.to("cuda"), targets.to("cuda")
      outputs = model(inputs)

      loss = criterion(outputs, targets)
      loss_valid += loss.item()
      _, predicted = torch.max(F.softmax(outputs, dim=1), 1)
      _, labels    = torch.max(targets, 1)
      correct += (predicted == labels).sum().item()

    writer.add_scalar("accuracy/valid", correct / len(dataloader_valid), epoch)

  writer.close()
