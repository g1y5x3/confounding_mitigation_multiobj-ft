import math
from typing import Any
import torch
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
from scipy.stats import norm
from torch.utils.data import Dataset, DataLoader

def num2vect(x, bin_range, bin_step, sigma):
    """
    v,bin_centers = number2vector(x,bin_range,bin_step,sigma)
    bin_range: (start, end), size-2 tuple
    bin_step: should be a divisor of |end-start|
    sigma:
    = 0 for 'hard label', v is index
    > 0 for 'soft label', v is vector
    < 0 for error messages.
    """
    bin_start = bin_range[0]
    bin_end = bin_range[1]
    bin_length = bin_end - bin_start
    if not bin_length % bin_step == 0:
      print("bin's range should be divisible by bin_step!")
      return -1
    bin_number = int(bin_length / bin_step)
    bin_centers = bin_start + float(bin_step) / 2 + bin_step * np.arange(bin_number)

    if sigma == 0:
      x = np.array(x)
      i = np.floor((x - bin_start) / bin_step)
      i = i.astype(int)
      return i, bin_centers
    elif sigma > 0:
      if np.isscalar(x):
        v = np.zeros((bin_number,))
        for i in range(bin_number):
          x1 = bin_centers[i] - float(bin_step) / 2
          x2 = bin_centers[i] + float(bin_step) / 2
          cdfs = norm.cdf([x1, x2], loc=x, scale=sigma)
          v[i] = cdfs[1] - cdfs[0]
        return v, bin_centers
      else:
        v = np.zeros((len(x), bin_number))
        for j in range(len(x)):
          for i in range(bin_number):
            x1 = bin_centers[i] - float(bin_step) / 2
            x2 = bin_centers[i] + float(bin_step) / 2
            cdfs = norm.cdf([x1, x2], loc=x[j], scale=sigma)
            v[j, i] = cdfs[1] - cdfs[0]
        return v, bin_centers

# NOTE All transform functions assume that input images are torch tensors
class CenterRandomShift(object):
  def __init__(self, randshift=None):
    self.shift = randshift

  def __call__(self, image):
    image_shape = torch.tensor(image.shape)
    center = image_shape // 2
    sub_shape = (image_shape // 8) * 8
    if self.shift:
      shift = torch.randint(-2,3,size=(3,))
      center = center + shift
    start = center - sub_shape // 2
    end = start + sub_shape
    return image[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

class RandomMirror(object):
  def __init__(self, dim=0, p=0.5):
    self.dim = dim
    self.p   = p

  def __call__(self, image):
    if torch.rand(1) > self.p:
      return image.flip(self.dim)
    else:
      return image

class IXIDataset(Dataset):
  def __init__(self, data_dir, label_file, sites=None, bin_range=None, transform=None):
    print(f"Loading file: {label_file}")
    self.directory = data_dir
    self.info = pd.read_csv(data_dir+"/"+label_file)
    self.transform = transform
    if not bin_range:
      self.bin_range = [math.floor(self.info['AGE'].min()), math.ceil(self.info['AGE'].max())]
      print(f"Age min {self.info['AGE'].min()}, Age max {self.info['AGE'].max()}")
      print("Computed Bin Range: ", self.bin_range)
    else:
      self.bin_range  = bin_range
      print(f"Provided Bin Range: {self.bin_range}")

    # Filter data based on specified sites
    if sites:
      if isinstance(sites, str):
        sites = [sites]
      self.info = self.info[self.info["SITE"].isin(sites)]
      print(f"Loading data from sites: {', '.join(sites)}")
    else:
      print("Loading data from all sites")
    #Reset index after filtering
    self.info = self.info.reset_index(drop=True)

    # Count the number of images from each site
    site_counts = self.info['SITE'].value_counts()
    print("Count of entries by SITE:")
    print(site_counts)

    total_count = site_counts.sum()
    print(f"\nTotal count for selected sites: {total_count}")

    # Pre-load the images and labels (if RAM is allowing)
    print(f"Filtered DataFrame shape: {self.info.shape}")
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

if __name__ == "__main__":
  bin_range   = [21,85]
  transform = [CenterRandomShift(randshift=True), RandomMirror()]
  data_train = IXIDataset(data_dir="data/IXI_4x4x4", label_file="IXI_train.csv", bin_range=bin_range, transform=transform)
  dataloader_train = DataLoader(data_train, batch_size=1, num_workers=0, pin_memory=True, shuffle=True)
  x, y = next(iter(dataloader_train))
  x, y = next(iter(dataloader_train))
  x, y = next(iter(dataloader_train))
  print(x.shape)
