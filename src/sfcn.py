import torch.nn as nn
import torch.nn.functional as F

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
