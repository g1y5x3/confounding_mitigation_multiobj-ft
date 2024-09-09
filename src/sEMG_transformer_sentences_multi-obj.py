import copy, torch, wandb, h5py, argparse
import numpy as np
import scipy.io as sio
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
from mlconfound.stats import partial_confound_test
from multiprocessing.pool import ThreadPool
from pymoo.optimize import minimize
from pymoo.operators.mutation.pm import PM
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.core.problem import StarmapParallelization
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.problem import ElementwiseProblem

def load_raw_signals(file):
  data = h5py.File(file, 'r')
  print(data.keys())

  signals = np.empty((40,4), dtype=object)
  labels = np.empty((40,1), dtype=object)
  vfi_1 = np.empty((40,1), dtype=object)
  sub_id = np.empty((40,1), dtype=object)
  sub_skinfold = np.empty((40,1), dtype=object)

  for i in range(40):
    for j in range(4):
      signals[i,j] = np.transpose(data[data['DATA'][j,i]])
    
    labels[i,0] = np.transpose(data[data['LABEL'][0,i]])

  print(f"signals {signals.shape}")
  print(f"labels {labels.shape}")

  # print(data['LABEL'])
  # print(data[data['LABEL'][0, 0]].shape)

  labels = np.array(data['LABEL'])
  vfi_1 = np.array(data['SUBJECT_VFI'])
  sub_id = np.array(data['SUBJECT_ID'])
  sub_skinfold = np.array(data['SUBJECT_SKINFOLD'])

  # print(labels.shape)
  # print(vfi_1.shape)
  # print(sub_id.shape)
  # print(sub_skinfold.shape)
 
  return signals, labels, vfi_1, sub_id, sub_skinfold

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

class sEMGtransformer(nn.Module):
  def __init__(self, patch_size=64, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1, num_layers=1):
    super().__init__()
    self.patch_size = patch_size
    self.seq_len = 4000 // patch_size

    self.input_project = nn.Linear(4*self.patch_size, d_model)
    self.dropout = nn.Dropout(dropout)
    encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                                               activation=nn.GELU(), batch_first=True, norm_first=True)
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    self.mlp_head = nn.Linear(d_model, 2)

    self.cls_token = nn.Parameter(torch.rand(1, 1, d_model))
    self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_len+1, d_model))

  def forward(self, x):
    # convert from raw signals to signal patches
    x = x[:, :, :(x.shape[2] // self.patch_size)*self.patch_size]
    B, C, L = x.shape
    x = x.reshape(B, C, L//self.patch_size, self.patch_size)  # [B, C, seq_len, patch_size]
    x = x.permute(0, 2, 1, 3).flatten(2,3)                    # [B, seq_len, C*patch_size]
    x = self.input_project(x)                                 # [B, seq_len, d_model]

    # add class token and positional embedding
    cls_token = self.cls_token.repeat(B,1,1)
    x = torch.cat((cls_token, x), dim=1)
    x = x + self.pos_embedding[:,:(self.seq_len+1)]
    x = self.dropout(x)

    x = self.transformer_encoder(x)

    # compare to using only the cls_token, using mean of embedding has a much smoother loss curve
    # x = x.mean(dim=1)
    x = x[:,0,:]
    x = self.mlp_head(x)
    return x

class OptimizeMLPLayer(ElementwiseProblem):
  def __init__(self, n_var=512, n_obj=2, n_constr=0, xl = -1*np.ones(512), xu = 1*np.ones(512), **kwargs):
    super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl = xl, xu = xu, **kwargs)

  def load_data(self, x_train, y_train, y_train_cpt, c_train, clf, perm):
    self.clf = clf.to("cuda")
    self.x_train = torch.tensor(x_train, dtype=torch.float, device="cuda")
    self.y_train = torch.tensor(y_train, dtype=torch.float, device="cuda")
    self.y_train_cpt = y_train_cpt
    self.criterion = nn.CrossEntropyLoss()
    self.c_train = c_train
    self.perm = perm

  def _evaluate(self, x, out, *args, **kwargs):
    with torch.no_grad():
      weight = torch.tensor(x.reshape((2,256)), dtype=torch.float, device="cuda")
      clf_copy = deepcopy(self.clf)
      clf_copy.mlp_head.weight.data = weight
      output = clf_copy(self.x_train)
      cross_entropy_loss = self.criterion(output, self.y_train)

      _, predicted = torch.max(F.softmax(output, dim=1), 1)
      y_pred_cpt = np.array(predicted.to("cpu"))
      ret = partial_confound_test(self.y_train_cpt, y_pred_cpt, self.c_train, cat_y=True, cat_yhat=True, cat_c=False, progress=False)

      out['F'] = [cross_entropy_loss.to("cpu").numpy(), 1-ret.p]

def count_correct(outputs, targets):
  _, predicted = torch.max(F.softmax(outputs, dim=1), 1)
  _, labels    = torch.max(targets, 1)
  return (predicted == labels).sum().item()

def train(config, signals, labels, sub_id, sub_skinfold):
  sub_test = config.sub_idx
  print(f"Subject R{sub_id[args.sub_idx][0][0][0]}")

  X, Y, C = [], [], []
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
    C.append(sub_skinfold[i][0].mean(axis=1))

  # normalize the signals channel-wise
  X_means = np.mean(np.concatenate(X, axis=0), axis=(0,2))
  X_stds = np.std(np.concatenate(X, axis=0), axis=(0,2))
  for i in range(40):
    X[i] = (X[i] - X_means[np.newaxis,:,np.newaxis]) / X_stds[np.newaxis,:,np.newaxis]
  print(f"X {np.concatenate(X, axis=0).shape}")

  # leave-one-out split
  X_test, Y_test = X[sub_test], Y[sub_test]
  X, Y, C = X[:sub_test] + X[sub_test+1:], Y[:sub_test] + Y[sub_test+1:], C[:sub_test] + C[sub_test+1:]
  X, Y, C = np.concatenate(X, axis=0), np.concatenate(Y, axis=0), np.concatenate(C, axis=0)

  num_samples = X.shape[0]
  indices = np.arange(num_samples)
  np.random.shuffle(indices)
  split_idx = int(num_samples*0.9)
  train_idx, valid_idx = indices[:split_idx], indices[split_idx:]

  X_train, X_valid = X[train_idx], X[valid_idx]
  Y_train, Y_valid = Y[train_idx], Y[valid_idx]
  Y_train_cpt = np.argmax(Y_train, axis=1)
  C_train = C[train_idx]
  print(f"X_train {X_train.shape}")
  print(f"X_valid {X_valid.shape}")
  print(f"X_test {X_test.shape}")

  dataset_train = sEMGSignalDataset(X_train, Y_train)
  dataset_valid = sEMGSignalDataset(X_valid, Y_valid)
  dataset_test  = sEMGSignalDataset(X_test, Y_test)

  dataloader_train     = DataLoader(dataset_train, batch_size=config.bsz, shuffle=True)
  dataloader_train_cpt = DataLoader(dataset_train, batch_size=config.bsz, shuffle=False)
  dataloader_valid     = DataLoader(dataset_valid, batch_size=config.bsz, shuffle=False)
  dataloader_test      = DataLoader(dataset_test,  batch_size=config.bsz, shuffle=False)

  model = sEMGtransformer(patch_size=config.psz, d_model=config.d_model, nhead=config.nhead, dim_feedforward=config.dim_feedforward,
                          dropout=config.dropout, num_layers=config.num_layers)
  model.to("cuda")

  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
  scaler = torch.cuda.amp.GradScaler()

  accuracy_train_best = 0
  accuracy_valid_best = 0
  accuracy_test_best = 0
  model_best = None
  for epoch in tqdm(range(config.epochs), desc="Training"):
    loss_train = 0
    correct_train = 0
    model.train()
    for batch, (inputs, targets) in enumerate(dataloader_train):
      inputs, targets = inputs.to("cuda"), targets.to("cuda")
      optimizer.zero_grad()
      with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = model(inputs)
        loss = criterion(outputs, targets)

      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()

      correct_train += count_correct(outputs, targets)
      loss_train += loss.item()

    wandb.log({"loss/train": loss_train/len(dataset_train), "accuracy/train": correct_train/len(dataset_train)}, step=epoch)

    loss_valid = 0
    correct_valid = 0
    model.eval()
    for inputs, targets in dataloader_valid:
      inputs, targets = inputs.to("cuda"), targets.to("cuda")
      outputs = model(inputs)
      loss = criterion(outputs, targets)
      correct_valid += count_correct(outputs, targets)
      loss_valid += loss.item()

    wandb.log({"loss/valid": loss_valid/len(dataset_valid), "accuracy/valid": correct_valid/len(dataset_valid)}, step=epoch)

    if correct_valid/len(dataset_valid) > accuracy_valid_best:
      accuracy_train_best = correct_train/len(dataset_train)
      accuracy_valid_best = correct_valid/len(dataset_valid)
      model_best = copy.deepcopy(model)
      correct_test = 0
      for inputs, targets in dataloader_test:
        inputs, targets = inputs.to("cuda"), targets.to("cuda")
        outputs = model(inputs)
        correct_test += count_correct(outputs, targets)
      accuracy_test_best = correct_test/len(dataset_test)
      wandb.log({"accuracy/test": correct_test/len(dataset_test)}, step=epoch)

    scheduler.step()

  Y_train_cpt = np.argmax(Y_train, axis=1)
  Y_pred = []
  for inputs, targets in dataloader_train_cpt:
    inputs, targets = inputs.to("cuda"), targets.to("cuda")
    outputs = model_best(inputs)
    _, predicted = torch.max(F.softmax(outputs, dim=1), 1)
    Y_pred.append(predicted.cpu().numpy())
  Y_pred_cpt = np.concatenate(Y_pred, axis=0)
  ret = partial_confound_test(Y_train_cpt, Y_pred_cpt, C_train, cat_y=True, cat_yhat=True, cat_c=False)

  print(f"accuracy_train_best: {accuracy_train_best}")
  print(f"P-value: {ret.p}")
  print(f"accuracy_valid_best: {accuracy_valid_best}")
  print(f"accuracy_test_best:  {accuracy_test_best}")

  wandb.log({"result/p-value": ret.p,
             "result/training": accuracy_train_best,
             "result/validation": accuracy_valid_best,
             "result/testing": accuracy_test_best})

  with torch.no_grad():
    print('Genetic Algorithm Optimization...')

    # test with boundries
    xl = np.ones(512) * model_best.mlp_head.weight.min().cpu().numpy()
    xu = np.ones(512) * model_best.mlp_head.weight.max().cpu().numpy()

    pool = ThreadPool(config.thread)
    runner = StarmapParallelization(pool.starmap)
    problem = OptimizeMLPLayer(elementwise_runner=runner)
    problem.load_data(X_train, Y_train, Y_train_cpt, C_train, model_best, config.perm)

    # Genetic algorithm initialization
    algorithm = NSGA2(pop_size  = config.pop,
                      sampling  = FloatRandomSampling(),
                      crossover = SBX(eta=15, prob=0.9),
                      mutation  = PM(eta=20),
                      output    = MultiObjectiveOutput())

    res = minimize(problem,
                   algorithm,
                   ("n_gen", config.ngen),
                   verbose=True)

    print('Completed! ', res.exec_time)
    pool.close()
    print(res.F)

    # Evaluate the results from GA optimization
    accuracy_train_best_cpt = 0
    accuracy_valid_best_cpt = 0
    accuracy_test_best_cpt = 0
    p_value_best_cpt = 0
    for i in range(len(res.X)):
      weight = torch.tensor(res.X[i,:].reshape((2,256)), dtype=torch.float32, device="cuda")
      model_copy = copy.deepcopy(model)
      model_copy.mlp_head.weight.data = weight
      model_copy.eval()

      # re-calcualte training accuracy and p-value
      correct_train = 0
      Y_pred = []
      for inputs, targets in dataloader_train_cpt:
        inputs, targets = inputs.to("cuda"), targets.to("cuda")
        outputs = model_copy(inputs)
        correct_train += count_correct(outputs, targets)
        _, predicted = torch.max(F.softmax(outputs, dim=1), 1)
        Y_pred.append(predicted.cpu().numpy())

      print(f"\nTraining Accuracy: {correct_train/len(dataset_train)}")
      Y_pred_cpt = np.concatenate(Y_pred, axis=0)
      ret = partial_confound_test(Y_train_cpt, Y_pred_cpt, C_train, cat_y=True, cat_yhat=True, cat_c=False, progress=False)
      print(f"P-value: {ret.p}")

      # re-calculate validation accuracy
      correct_valid = 0
      for inputs, targets in dataloader_valid:
        inputs, targets = inputs.to("cuda"), targets.to("cuda")
        outputs = model_copy(inputs)
        correct_valid += count_correct(outputs, targets)

      print(f"Validation Accuracy: {correct_valid/len(dataset_valid)}")

      # re-calculate testing accuracy
      correct_test = 0
      for inputs, targets in dataloader_test:
        inputs, targets = inputs.to("cuda"), targets.to("cuda")
        outputs = model_copy(inputs)
        correct_test += count_correct(outputs, targets)

      print(f"Testing Accuracy: {correct_test/len(dataset_test)}")

      if correct_test/len(dataset_test) > accuracy_test_best_cpt:
        accuracy_test_best_cpt = correct_test/len(dataset_test)
        accuracy_train_best_cpt = correct_train/len(dataset_train)
        accuracy_valid_best_cpt = correct_valid/len(dataset_valid)
        p_value_best_cpt = ret.p

  wandb.log({"result/p-value-cpt": p_value_best_cpt,
             "result/training-cpt": accuracy_train_best_cpt,
             "result/validation-cpt": accuracy_valid_best_cpt,
             "result/testing-cpt": accuracy_test_best_cpt})


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="sEMG transformer training configurations")
  # experiment config
  parser.add_argument('--sub_idx', type=int, default=0, help="subject index")
  # training config
  parser.add_argument('--seed', type=int, default=0, help="random seed")
  parser.add_argument('--epochs', type=int, default=500, help="number of epochs")
  parser.add_argument('--bsz', type=int, default=64, help="batch size")
  # optimizer config
  parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
  parser.add_argument('--wd', type=float, default=0.001, help="weight decay")
  parser.add_argument('--step_size', type=int, default=500, help="lr scheduler step size")
  parser.add_argument('--gamma', type=float, default=0.8, help="lr scheduler gamma")
  # model config
  parser.add_argument('--psz', type=int, default=64, help="signal patch size")
  parser.add_argument('--d_model', type=int, default=256, help="transformer embedding dim")
  parser.add_argument('--nhead', type=int, default=8, help="transformer number of attention heads")
  parser.add_argument('--dim_feedforward', type=int, default=1024, help="transformer feed-forward dim")
  parser.add_argument('--num_layers', type=int, default=3, help="number of transformer encoder layers")
  parser.add_argument('--dropout', type=float, default=0.3, help="dropout rate")
  # genetic algorithm config
  parser.add_argument('--ngen', type=int, default=4, help="Number of generation")
  parser.add_argument('--pop', type=int, default=32, help='Population size')
  parser.add_argument('--thread', type=int, default=4, help='Number of threads')
  parser.add_argument('--perm', type=int, default=100, help='Permutation value')
  args = parser.parse_args()

  # load data
  signals, labels, vfi_1, sub_id, sub_skinfold = load_raw_signals("data/subjects_40_sen_raw.mat")
  # (40, 4)
  # <class 'numpy.ndarray'>
  # (165, 4000)
  # (40, 1)
  # <class 'numpy.ndarray'>
  # (40, 1)
  # <class 'numpy.ndarray'>
  # (40, 1)
  # <class 'numpy.ndarray'>
  # (40, 1)
  # <class 'numpy.ndarray'>  

  wandb.init(project="sEMG_transformers", name=f"R{sub_id[args.sub_idx][0][0][0]}", config=args)
  config = wandb.config

  np.random.seed(config.seed)
  torch.manual_seed(config.seed)

  train(config, signals, labels, sub_id, sub_skinfold)

  wandb.finish()