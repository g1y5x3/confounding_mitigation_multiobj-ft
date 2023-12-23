import os
import wandb
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader

from torchinfo import summary
from tqdm import trange
from sfcn import SFCN
from imageloader import IXIDataset, CenterRandomShift, RandomMirror

WANDB = os.getenv("WANDB", False)
GROUP = os.getenv("GROUP", "tests")
NAME  = os.getenv("NAME" , "test")

DATA_DIR   = os.getenv("DATA_DIR",   "data/IXI_4x4x4")
DATA_SPLIT = os.getenv("DATA_SPLIT", "all")

def train(config, run=None):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  print(config) 

  # TODO: Test with other range that does not produce a x64 output
  bin_range   = [21,85]

  print("\nDataloader:")
  # based on the paper the training inputs are 
  # 1) randomly shifted by 0, 1, or 2 voxels along every axis; 
  # 2) has a probability of 50% to be mirrored about the sagittal plane
  data_train = IXIDataset(data_dir=DATA_DIR, label_file=f"IXI_{DATA_SPLIT}_train.csv", bin_range=bin_range, transform=[CenterRandomShift(randshift=True), RandomMirror()])
  data_test  = IXIDataset(data_dir=DATA_DIR, label_file=f"IXI_{DATA_SPLIT}_test.csv",  bin_range=bin_range, transform=[CenterRandomShift(randshift=False)])
  bin_center = data_train.bin_center.reshape([-1,1])

  dataloader_train = DataLoader(data_train, batch_size=config["batch_size"], num_workers=config["num_workers"], pin_memory=True, shuffle=True)
  dataloader_test  = DataLoader(data_test,  batch_size=config["batch_size"], num_workers=config["num_workers"], pin_memory=True, shuffle=False)
  
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
  # load pretrained weights from https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain/raw/master/brain_age/run_20190719_00_epoch_best_mae.p
  w_pretrained = torch.load("model/run_20190719_00_epoch_best_mae.p")
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

  t = trange(config["num_epochs"], desc="\nTraining", leave=True)
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
  
    scheduler.step()
  
    t.set_description(f"Training: train/loss {loss_train:.2f}, train/MAE_age {MAE_age_train:.2f} test/loss {loss_test:.2f}, test/MAE_age {MAE_age_test:.2f}")
    if run:
      wandb.log({"train/loss": loss_train,
                 "train/MAE_age": MAE_age_train,
                 "test/loss":  loss_test,
                 "test/MAE_age":  MAE_age_test,
                 })
  
  # Save and upload the trained model 
  torch.save(model.state_dict(), "model/model.pth")
  if run:
    artifact = wandb.Artifact("model", type="model")
    artifact.add_file("model/model.pth")
    run.log_artifact(artifact)
    run.finish()

  return loss_test, MAE_age_test

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="Example:")
  parser.add_argument("--batch_size",  type=int,   default=8,    help="batch size")
  parser.add_argument("--num_workers", type=int,   default=2,    help="number of workers")
  parser.add_argument("--num_epochs",  type=int,   default=10,   help="total number of epochs")
  parser.add_argument("--lr",          type=float, default=1e-2, help="learning rate")
  parser.add_argument("--wd",          type=float, default=1e-3, help="weight decay")
  parser.add_argument("--step_size",   type=int,   default=30,   help="step size")
  parser.add_argument("--gamma",       type=float, default=0.3,  help="gamma")
  args = parser.parse_args()
  config = vars(args)

  if WANDB:
    # TODO need to pass project/group without using argparse
    run = wandb.init(
      project = "Confounding-in-fMRI-Deep-Learning",
      name    = NAME,
      group   = GROUP,
      config  = config
    )
  else:
    run = None
  
  train(config, run)
