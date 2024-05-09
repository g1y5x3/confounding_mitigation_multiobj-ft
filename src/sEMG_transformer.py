import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from util.sEMGhelpers import load_raw_signals
from util.sEMGFeatureLoader import sEMGSignalDataset

class sEMGtransformer(nn.Module):
  def __init__(self, patch_size=256, d_model=512, nhead=8, dim_feedforward=2048):
    super().__init__()
    self.patch_size = patch_size
    self.d_model = d_model
    self.seq_len = 4000 // patch_size
    self.input_project = nn.Linear(4*self.patch_size, d_model)
    self.encoder = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                              dropout=0.1, activation=nn.GELU(), batch_first=True, norm_first=True)
    self.output_project = nn.Linear(d_model, 2)

    # Parameters/Embeddings
    self.cls_token = nn.Parameter(torch.rand(1, 1, d_model))
    self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_len+1, d_model))

  def forward(self, x):
    # Convert from signal to patch
    x = x[:, :, :(x.shape[2] // self.patch_size)*self.patch_size]
    B, C, L = x.shape
    x = x.reshape(B, C, L//self.patch_size, self.patch_size)
    x = x.permute(0, 2, 1, 3)
    x = x.flatten(2,3)
    x = self.input_project(x)

    # Add class token and positional embedding
    cls_token = self.cls_token.repeat(B,1,1)
    x = torch.cat((cls_token, x), dim=1)
    x = x + self.pos_embedding[:,:(self.seq_len+1)]
    x = F.dropout(x, p=0.1)

    # Apply transformer
    x = self.encoder(x)
    x = x.mean(dim=1)
    return self.output_project(x)

if __name__ == "__main__":
  # signal pre-processing
  signals, labels, VFI1, sub_id, sub_skinfold = load_raw_signals("data/subjects_40_v6.mat")

  X, Y = [], []
  for i in range(40):
    # stack all inputs into [N,C,L] format
    x = np.stack(signals[i], axis=1)

    # one-hot encode the binary labels
    N = labels[i][0].shape[0]
    mapped_indices = (labels[i][0] == 1).astype(int)
    y_onehot = np.zeros((N, 2))
    y_onehot[np.arange(N), mapped_indices.flatten()] = 1

    X.append(x)
    Y.append(y_onehot)

  X, Y = np.concatenate(X, axis=0), np.concatenate(Y, axis=0)
  print(f"X {X.shape}")
  print(f"Y {Y.shape}")

  # normalize X channel-wise
  X_means = np.mean(X, axis=(0,2))
  X_stds = np.std(X, axis=(0,2))
  print(f"X means {X_means}")
  print(f"X stds {X_stds}")
  X_norm = (X - X_means[np.newaxis,:,np.newaxis]) / X_stds[np.newaxis,:,np.newaxis]

  # shuffle indices
  num_samples = X_norm.shape[0]
  indices = np.arange(num_samples)
  np.random.shuffle(indices)

  # split indices for train and test
  split_idx = int(num_samples*0.9)
  train_idx, valid_idx = indices[:split_idx], indices[split_idx:]

  X_train, X_valid = X_norm[train_idx], X_norm[valid_idx]
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

  model = sEMGtransformer(patch_size=32, d_model=512, nhead=8, dim_feedforward=2048)
  model.to("cuda")

  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.AdamW(model.parameters())

  writer = SummaryWriter()
  for epoch in tqdm(range(750), desc="Training Epochs"):
    loss_train = 0
    correct_train = 0
    model.train()
    for batch, (inputs, targets) in enumerate(dataloader_train):
      optimizer.zero_grad()

      inputs, targets = inputs.to("cuda"), targets.to("cuda")
      outputs = model(inputs)
      loss = criterion(outputs, targets)

      loss.backward()
      optimizer.step()

      _, predicted = torch.max(F.softmax(outputs, dim=1), 1)
      _, labels    = torch.max(targets, 1)
      correct_train += (predicted == labels).sum().item()
      loss_train += loss.item()

    writer.add_scalar("loss/train", loss_train/len(dataset_train), epoch)
    writer.add_scalar("accuracy/train", correct_train/len(dataset_train), epoch)

    loss_valid = 0
    correct_valid = 0
    model.eval()
    for inputs, targets in dataloader_valid:
      inputs, targets = inputs.to("cuda"), targets.to("cuda")
      outputs = model(inputs)
      loss = criterion(outputs, targets)

      _, predicted = torch.max(F.softmax(outputs, dim=1), 1)
      _, labels    = torch.max(targets, 1)
      correct_valid += (predicted == labels).sum().item()
      loss_valid += loss.item()

    writer.add_scalar("loss/valid", loss_valid/len(dataset_valid), epoch)
    writer.add_scalar("accuracy/valid", correct_valid/len(dataset_valid), epoch)

  writer.close()
