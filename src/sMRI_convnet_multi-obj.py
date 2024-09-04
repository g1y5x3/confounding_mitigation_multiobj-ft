import os, math, wandb, argparse, requests, torch
import numpy as np
import pandas as pd
import nibabel as nib
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
from mlconfound.stats import partial_confound_test
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy import stats
from util.fMRIImageLoader import num2vect, CenterRandomShift, RandomMirror
from multiprocessing.pool import ThreadPool
from pymoo.optimize import minimize
from pymoo.operators.mutation.pm import PM
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.core.problem import StarmapParallelization
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.problem import ElementwiseProblem

DATA_DIR   = os.getenv("DATA_DIR",   "data/IXI_4x4x4")
DATA_SPLIT = os.getenv("DATA_SPLIT", "all")

def generate_wandb_name(config):
  train_sites = '_'.join(sorted(config['site_train']))
  test_sites = '_'.join(sorted(config['site_test']))
  return f"train_{train_sites}_test_{test_sites}"

def create_confounded_sample(df, n_samples):
  # Convert site to numeric (0 for Guy's, 1 for Hammersmith's)
  df['SITE_INDEX'] = (df['SITE'] == "HH").astype(int)
  
  # Fit linear regression
  reg = LinearRegression().fit(df['SITE_INDEX'].to_numpy().reshape(-1,1), df['AGE'].to_numpy().reshape(-1,1))

  # calculate the age prediciton based on the site
  df['AGE_PRED'] = reg.predict(df['SITE_INDEX'].to_numpy().reshape(-1,1))
  df['RESIDUAL'] = df['AGE'] - df['AGE_PRED']
  
  df_sorted = df.sort_values('RESIDUAL', key=abs)
  sampled_df = df_sorted.head(n_samples)
  
  return sampled_df

def check_significance(df):
  guys_ages = df[df['SITE'] == "Guys"]['AGE']
  hammersmith_ages = df[df['SITE'] == "HH"]['AGE']
  t_stat, p_value = stats.ttest_ind(guys_ages, hammersmith_ages)
  return p_value < 0.05, p_value, t_stat

def load_and_split_data(config):
  df = pd.read_csv("data/IXI_all.csv")

  #filter out subjects whos below 47 years old  
  df = df[df["AGE"] > 47]

  df_train_val = df[df["SITE"].isin(config["site_train"])]
  df_test = df[df["SITE"].isin(config["site_test"])]

  # split the training and validation into half and half
  df_train, df_val = train_test_split(df_train_val, test_size=0.5, random_state=config["seed"])
  df_train_confounded = create_confounded_sample(df_train, n_samples=80)

  # Check significance
  is_significant, p_value, t_stat = check_significance(df_train_confounded)
  print(f"Confounded sample created: {is_significant}. p-value: {p_value:.4f}, t-statistic: {t_stat:.4f}")

  print(f"Training size: {len(df_train)}")
  print(f"validation size: {len(df_val)}")
  print(f"Testing size: {len(df_test)}")

  return df_train_confounded, df_val, df_test, p_value

class IXIDataset(Dataset):
  def __init__(self, data_dir, data_df, bin_range=None, transform=None):
    self.directory = data_dir
    self.info = data_df
    self.info = self.info.reset_index(drop=True)
    self.transform = transform

    if not bin_range:
      self.bin_range = [math.floor(self.info['AGE'].min()), math.ceil(self.info['AGE'].max())]
      print(f"Age min {self.info['AGE'].min()}, Age max {self.info['AGE'].max()}")
      print("Computed Bin Range: ", self.bin_range)
    else:
      self.bin_range  = bin_range
      print(f"Provided Bin Range: {self.bin_range}")

    # Count the number of images from each site
    site_counts = self.info['SITE'].value_counts()
    print("Count of entries by SITE:")
    print(site_counts)

    total_count = site_counts.sum()
    print(f"\nTotal count for selected sites: {total_count}")

    # Pre-load the images and labels (if RAM is allowing)
    print(self.info["FILENAME"][0])
    nii = nib.load(self.directory+"/"+self.info["FILENAME"][0])
    voxel_size = nii.header.get_zooms()
    print(f"Voxel Size: {voxel_size}")
    image = torch.tensor(nii.get_fdata(), dtype=torch.float32)
    self.image_all = torch.empty((len(self.info),) + tuple(image.shape), dtype=torch.float32)

    age = np.array([71.3])
    y, bc = num2vect(age, self.bin_range, 1, 1)
    label = torch.tensor(y, dtype=torch.float32)
    self.label_all = torch.empty((len(self.info),) + tuple(label.shape)[1:], dtype=torch.float32)

    for i in tqdm(range(len(self.info)), desc="Loading Data"):
      nii = nib.load(self.directory+"/"+self.info["FILENAME"][i])
      self.image_all[i,:] = torch.tensor(nii.get_fdata(), dtype=torch.float32)

      age = self.info["AGE"][i]
      y, _ = num2vect(age, self.bin_range, 1, 1)
      y += 1e-16
      self.label_all[i,:] = torch.tensor(y, dtype=torch.float32)

    self.bin_center = torch.tensor(bc, dtype=torch.float32)

    print(f"Image Dim {self.image_all.shape}")
    print(f"Label Dim {self.label_all.shape}")
    print(f"Min={self.image_all.min()}, Max={self.image_all.max()}, Mean={self.image_all.mean()}, Std={self.image_all.std()}")

  def __len__(self):
    return len(self.info)

  def __getitem__(self, idx):
    image, label = self.image_all[idx,:], self.label_all[idx,:]
    if self.transform:
      for tsfrm in self.transform:
        image = tsfrm(image)
    image = torch.unsqueeze(image, 0)
    return image, label

class SFCN(nn.Module):
  def __init__(self, channel_number=[32, 64, 128, 256, 256, 64], output_dim=40, dropout=True):
    super(SFCN, self).__init__()
    n_layer = len(channel_number)

    self.feature_extractor = nn.Sequential()
    for i in range(n_layer):
      in_channel = 1 if i == 0 else channel_number[i-1] 
      out_channel = channel_number[i]
      if i < n_layer-1:
        self.feature_extractor.add_module(f"conv_{i}",
                                          self.conv_layer(in_channel,
                                                          out_channel,
                                                          maxpool=True,
                                                          kernel_size=3,
                                                          padding=1))
      else:
        self.feature_extractor.add_module(f"conv_{i}",
                                          self.conv_layer(in_channel,
                                                          out_channel,
                                                          maxpool=False,
                                                          kernel_size=1,
                                                          padding=0))

    self.classifier = nn.Sequential()
    # NOTE initial model uses a average pool here
    if dropout: self.classifier.add_module('dropout', nn.Dropout(0.5))
    i = n_layer
    # TODO calculate or ask user to provide the dim size of handcoding it
    # otherwise this would have to change depends on the input image size
    in_channel = channel_number[-1]*2*2*2
    out_channel = output_dim
    self.classifier.add_module(f"fc_{i}", nn.Linear(in_channel, out_channel))

  @staticmethod
  def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2):
    if maxpool is True:
      layer = nn.Sequential(
        nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
        nn.BatchNorm3d(out_channel),
        nn.MaxPool3d(2, stride=maxpool_stride),
        nn.ReLU(),
      )
    else:
      layer = nn.Sequential(
        nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
        nn.BatchNorm3d(out_channel),
        nn.ReLU()
      )
    return layer

  def forward(self, x):
    x = self.feature_extractor(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    x = F.softmax(x, dim=1)
    return x

class OptimizeMLPLayer(ElementwiseProblem):
  def __init__(self, n_var=32768, n_obj=2, n_constr=0, xl = -1*np.ones(32768), xu = 1*np.ones(32768), **kwargs):
    super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl = xl, xu = xu, **kwargs)

  def load_data(self, y_train_cpt, c_train, dataloader, clf, bin_center, device, perm):
    self.clf = clf.to("cuda")
    self.bin_center = bin_center
    self.dataloader = dataloader
    self.y_train_cpt = y_train_cpt
    self.criterion = nn.KLDivLoss(reduction="batchmean", log_target=True)
    self.c_train = c_train
    self.device = device
    self.perm = perm

  def _evaluate(self, x, out, *args, **kwargs):
    with torch.no_grad():
      weight = torch.tensor(x.reshape((64,512)), dtype=torch.float, device="cuda")
      clf_copy = deepcopy(self.clf)
      clf_copy.classifier.fc_6.weight.data = weight

      loss_train = 0.0
      Y_predict = []
      for images, labels in self.dataloader:
        images, labels = images.to(self.device), labels.to(self.device)
        output = clf_copy(images)

        loss = self.criterion(output.log(), labels.log())
        age_pred   = output @ self.bin_center

        loss_train += loss.item()
        Y_predict.append(age_pred.cpu().numpy())

      loss_train = loss_train / len(self.dataloader)

      Y_predict = np.concatenate(Y_predict).squeeze()

      ret = partial_confound_test(self.y_train_cpt, Y_predict, self.c_train, cat_y=False, cat_yhat=False, cat_c=True, progress=False)

      out['F'] = [loss_train, 1-ret.p]

def train(config, run=None):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  print(config) 

  # TODO: Test with other range that does not produce a x64 output
  bin_range   = [21,85]

  print("\nDataloader:")
  # based on the paper the training inputs are 
  # 1) randomly shifted by 0, 1, or 2 voxels along every axis; 
  # 2) has a probability of 50% to be mirrored about the sagittal plane
  df_train, df_val, df_test, p_t_test = load_and_split_data(config)
  wandb.run.summary["results/p_value_t-test"] = p_t_test

  data_train = IXIDataset(data_dir=DATA_DIR, data_df=df_train,
                          bin_range=bin_range, 
                          transform=[CenterRandomShift(randshift=True), RandomMirror()])

  data_valid = IXIDataset(data_dir=DATA_DIR, data_df=df_val,
                          bin_range=bin_range, 
                          transform=[CenterRandomShift(randshift=False)])

  data_test  = IXIDataset(data_dir=DATA_DIR, data_df=df_test,  
                          bin_range=bin_range,
                          transform=[CenterRandomShift(randshift=False)])

  bin_center = data_train.bin_center.reshape([-1,1])

  dataloader_train     = DataLoader(data_train, batch_size=config["bs"], num_workers=config["num_workers"], pin_memory=True, shuffle=True)
  dataloader_train_cpt = DataLoader(data_train, batch_size=config["bs"], num_workers=config["num_workers"], pin_memory=True, shuffle=False)
  dataloader_valid     = DataLoader(data_valid, batch_size=config["bs"], num_workers=config["num_workers"], pin_memory=True, shuffle=False)
  dataloader_test      = DataLoader(data_test,  batch_size=config["bs"], num_workers=config["num_workers"], pin_memory=True, shuffle=False)
  
  x, y = next(iter(dataloader_train))
  print("\nTraining data summary:")
  print(f"Total data: {len(data_train)}")
  print(f"Input {x.shape}")
  print(f"Label {y.shape}")
  
  x, y = next(iter(dataloader_test))
  print("\nTesting data summary:")
  print(f"Total data: {len(data_test)}")
  print(f"Input {x.shape}")
  print(f"Label {y.shape}")
  
  model = SFCN(output_dim=y.shape[1])
  print(f"\nModel Dtype: {next(model.parameters()).dtype}")
  summary(model, x.shape)

  # load pretrained weights shared by the original author
  url = "https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain/raw/master/brain_age/run_20190719_00_epoch_best_mae.p"
  filename = "run_20190719_00_epoch_best_mae.pth" 
  if not os.path.exists(filename):
    response = requests.get(url)
    with open("run_20190719_00_epoch_best_mae.pth", "wb") as file:
      file.write(response.content)

  w_pretrained = torch.load("run_20190719_00_epoch_best_mae.pth")
  w_feature_extractor = {k: v for k, v in w_pretrained.items() if "module.classifier" not in k}
  model.load_state_dict(w_feature_extractor, strict=False)
  
  criterion = nn.KLDivLoss(reduction="batchmean", log_target=True)
  optimizer = optim.SGD(model.parameters(), lr=config["lr"], weight_decay=config["wd"])
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["step_size"], gamma=config["gamma"])
  scaler = torch.cuda.amp.GradScaler(enabled=True)
  
  # main training loop
  print(criterion)
  model.to(device)
  bin_center = bin_center.to(device)

  MAE_age_train_best = float('inf')
  MAE_age_valid_best = float('inf')
  MAE_age_test_best = float('inf')
  model_best = None
  
  t = trange(config["epochs"], desc="\nTraining", leave=True)
  for epoch in t:
    loss_train = 0.0
    MAE_age_train = 0.0
    for images, labels in dataloader_train:
      images, labels = images.to(device), labels.to(device)
      with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        output = model(images)
        loss = criterion(output.log(), labels.log())
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()
      optimizer.zero_grad()
  
      with torch.no_grad():
        age_target = labels @ bin_center
        age_pred   = output @ bin_center
        MAE_age = F.l1_loss(age_pred, age_target, reduction="mean")
  
        loss_train += loss.item()
        MAE_age_train += MAE_age.item()
  
    loss_train = loss_train / len(dataloader_train)
    MAE_age_train = MAE_age_train / len(dataloader_train)
  
    with torch.no_grad():
      loss_valid = 0.0
      MAE_age_valid = 0.0
      for images, labels in dataloader_valid:
        x, y = images.to(device), labels.to(device)
        output = model(x)
        loss = criterion(output.log(), y.log())
  
        age_target = y @ bin_center
        age_pred   = output @ bin_center
        MAE_age = F.l1_loss(age_pred, age_target, reduction="mean")
  
        loss_valid += loss.item()
        MAE_age_valid += MAE_age.item()

      loss_valid = loss_valid / len(dataloader_valid)
      MAE_age_valid = MAE_age_valid / len(dataloader_valid)

      loss_test = 0.0
      MAE_age_test = 0.0
      for images, labels in dataloader_test:
        x, y = images.to(device), labels.to(device)
        output = model(x)
        loss = criterion(output.log(), y.log())
  
        age_target = y @ bin_center
        age_pred   = output @ bin_center
        MAE_age = F.l1_loss(age_pred, age_target, reduction="mean")
  
        loss_test += loss.item()
        MAE_age_test += MAE_age.item()
 
    loss_test = loss_test / len(dataloader_test)
    MAE_age_test = MAE_age_test / len(dataloader_test)

    if MAE_age_test < MAE_age_test_best:
      MAE_age_train_best = MAE_age_train
      MAE_age_valid_best = MAE_age_valid
      MAE_age_test_best = MAE_age_test
      model_best = deepcopy(model)
  
    scheduler.step()
  
    t.set_description(f"Training: train/MAE_age {MAE_age_train:.2f} valid/MAE_age {MAE_age_valid:.2f}, test/MAE_age {MAE_age_test:.2f}")

    wandb.log({"train/loss": loss_train,
               "train/MAE_age": MAE_age_train,
               "valid/loss": loss_valid,
               "valid/MAE_age": MAE_age_valid,
               "test/loss":  loss_test,
               "test/MAE_age":  MAE_age_test,
               })

  # Running the initial confounding test   
  with torch.no_grad():
    # Convert the SITE from text to indices
    site_index, _ = pd.factorize(df_train['SITE'])
    print(site_index)
    C, Y_target, Y_predict = np.array(site_index), [], []
    loss_train = 0
    MAE_age_train = 0
    for images, labels in dataloader_train_cpt:
      images, labels = images.to(device), labels.to(device)
      output = model_best(images)

      loss = criterion(output.log(), labels.log())
      loss_train += loss.item()

      age_target = labels @ bin_center
      age_pred   = output @ bin_center
      MAE_age = F.l1_loss(age_pred, age_target, reduction="mean")
      MAE_age_train += MAE_age.item()

      Y_target.append(age_target.cpu().numpy())
      Y_predict.append(age_pred.cpu().numpy())

    loss_train = loss_train / len(dataloader_train_cpt)
    MAE_age_train = MAE_age_train / len(dataloader_train_cpt)

    Y_target = np.concatenate(Y_target).squeeze()
    Y_predict = np.concatenate(Y_predict).squeeze()

    print(f"C: {C.shape}")
    print(f"Y_target: {Y_target.shape}")
    print(f"Y_predict: {Y_predict.shape}")
      
    # Run conditional permutation test
    ret = partial_confound_test(Y_target, Y_predict, C, cat_y=False, cat_yhat=False, cat_c=True, progress=True)
    print(f"MAE_age_train: {MAE_age_train_best}")
    print(f"MAE_age_valid: {MAE_age_valid_best}")
    print(f"MAE_age_test:  {MAE_age_test_best}")
    print(f"P-value: {ret.p}")

    wandb.run.summary["results/MAE_age_train"] = MAE_age_train_best
    wandb.run.summary["results/MAE_age_valid"] = MAE_age_valid_best
    wandb.run.summary["results/MAE_age_test"] = MAE_age_test_best
    wandb.run.summary["results/p_value"] = ret.p

    print('Genetic Algorithm Optimization...')

    # test with boundries for weights search
    lower_bound = model.classifier.fc_6.weight.min().cpu().numpy() * 5
    upper_bound = model.classifier.fc_6.weight.max().cpu().numpy() * 5
    wandb.log({
      "xl": lower_bound,
      "xu": upper_bound
    })

    xl = np.ones(32768) * lower_bound 
    xu = np.ones(32768) * upper_bound

    pool = ThreadPool(config.thread)
    runner = StarmapParallelization(pool.starmap)
    problem = OptimizeMLPLayer(elementwise_runner=runner, xl=xl, xu=xu)
    problem.load_data(Y_target, C, dataloader_train_cpt, model_best, bin_center, device, config.perm)

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

    MAE_age_train_best_cpt = float('inf')
    MAE_age_valid_best_cpt = float('inf')
    MAE_age_test_best_cpt = float('inf')
    p_value_best_cpt = 0
    
    for i in range(len(res.X)):
      weight = torch.tensor(res.X[i,:].reshape((64,512)), dtype=torch.float32, device="cuda")
      model_copy = deepcopy(model)
      model_copy.classifier.fc_6.weight.data = weight
      model_copy.eval()

      Y_predict = []
      loss_train = 0
      MAE_age_train = 0
      for images, labels in dataloader_train_cpt:
        images, labels = images.to(device), labels.to(device)
        output = model_copy(images)
        loss = criterion(output.log(), labels.log())
        loss_train += loss.item()

        age_target = labels @ bin_center
        age_pred   = output @ bin_center
        MAE_age = F.l1_loss(age_pred, age_target, reduction="mean")
        MAE_age_train += MAE_age.item()

        Y_predict.append(age_pred.cpu().numpy())

      loss_train = loss_train / len(dataloader_train_cpt)
      MAE_age_train = MAE_age_train / len(dataloader_train_cpt)

      Y_predict = np.concatenate(Y_predict).squeeze()

      ret = partial_confound_test(Y_target, Y_predict, C, cat_y=False, cat_yhat=False, cat_c=True, progress=True)
      
      loss_valid = 0.0
      MAE_age_valid = 0.0
      for images, labels in dataloader_valid:
        x, y = images.to(device), labels.to(device)
        output = model_copy(x)
        loss = criterion(output.log(), y.log())
  
        age_target = y @ bin_center
        age_pred   = output @ bin_center
        MAE_age = F.l1_loss(age_pred, age_target, reduction="mean")
  
        loss_valid += loss.item()
        MAE_age_valid += MAE_age.item()

      loss_valid = loss_valid / len(dataloader_valid)
      MAE_age_valid = MAE_age_valid / len(dataloader_valid)

      loss_test = 0.0
      MAE_age_test = 0.0
      for images, labels in dataloader_test:
        x, y = images.to(device), labels.to(device)
        output = model_copy(x)
        loss = criterion(output.log(), y.log())
  
        age_target = y @ bin_center
        age_pred   = output @ bin_center
        MAE_age = F.l1_loss(age_pred, age_target, reduction="mean")
  
        loss_test += loss.item()
        MAE_age_test += MAE_age.item()

      loss_test = loss_test / len(dataloader_test)
      MAE_age_test = MAE_age_test / len(dataloader_test)

      print(f"P-value: {ret.p}")
      print(f"MAE_age_train {MAE_age_train}")
      print(f"MAE_age_valid {MAE_age_valid}")
      print(f"MAE_age_test {MAE_age_test}")

      if MAE_age_valid < MAE_age_valid_best_cpt:
        MAE_age_train_best_cpt = MAE_age_train
        MAE_age_valid_best_cpt = MAE_age_valid
        MAE_age_test_best_cpt = MAE_age_test
        p_value_best_cpt = ret.p

  wandb.run.summary["results/MAE_age_train_cpt"] = MAE_age_train_best_cpt
  wandb.run.summary["results/MAE_age_valid_cpt"] = MAE_age_valid_best_cpt
  wandb.run.summary["results/MAE_age_test_cpt"] = MAE_age_test_best_cpt
  wandb.run.summary["results/p_value_cpt"] = p_value_best_cpt

  # Save and upload the trained model 
  torch.save(model.state_dict(), "model.pth")

  artifact = wandb.Artifact("model", type="model")
  artifact.add_file("model.pth")
  run.log_artifact(artifact)

  print(f"\nTraining completed. Best MAE_age achieved: {MAE_age_test_best:.4f}")

  run.finish()

  return loss_test, MAE_age_test

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="Example:")
  # specify training and testing site
  parser.add_argument("--site_train", nargs='+', default=["Guys", "HH"], 
                      help="List of sites for training data (e.g., --site_train Guys HH)")
  parser.add_argument("--site_test", nargs='+', default=["IOP"], 
                      help="List of sites for testing data (e.g., --site_test IOP)")
  # neural network config
  parser.add_argument("--bs", type=int,   default=8,    help="batch size")
  parser.add_argument("--num_workers", type=int,   default=2,    help="number of workers")
  parser.add_argument("--epochs", type=int,   default=10,   help="total number of epochs")
  parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
  parser.add_argument("--wd", type=float, default=1e-3, help="weight decay")
  parser.add_argument("--step_size", type=int,   default=30,   help="step size")
  parser.add_argument("--gamma", type=float, default=0.3,  help="gamma")
  # genetic algorithm config
  parser.add_argument('--ngen', type=int, default=2, help="Number of generation")
  parser.add_argument('--pop', type=int, default=32, help='Population size')
  parser.add_argument('--thread', type=int, default=4, help='Number of threads')
  parser.add_argument('--perm', type=int, default=100, help='Permutation value')
  # random seed
  parser.add_argument('--seed', type=int, default=62, help="random seed for sampling subjects")
  args = parser.parse_args()

  wandb_name = generate_wandb_name(vars(args))

  run = wandb.init(project="fMRI-ConvNets", name=wandb_name, config=args)
  config = run.config

  train(config, run)
