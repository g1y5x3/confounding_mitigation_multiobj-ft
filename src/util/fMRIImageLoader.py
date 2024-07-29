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
