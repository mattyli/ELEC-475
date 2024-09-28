import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torchvision.transforms.functional as F
from torchinfo import summary

KERNEL_SIZE = 3     # kernel size for maxpool and conv
STRIDE = 2          # stride for maxpool and conv
PADDING = 1         # padding for maxpool and conv
DROPOUT_P = 0.0     # dropout probability

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SnoutNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.featureNet = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING),     # CONV 1
            nn.ReLU(),                                                                                             
            nn.MaxPool2d(kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING),                                  # MAX 1
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING),   # CONV2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING),                                  # MAX 2
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING),  # CONV 3
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING),                                  # MAX 3
            nn.ReLU()
        )

        self.regressor = nn.Sequential(
            nn.Dropout(p=DROPOUT_P, inplace=True),
            nn.Linear(in_features=256*4*4, out_features=1024),      # (FC1) b x 256 x 4 x 4
            nn.ReLU(),      
            nn.Dropout(p=DROPOUT_P, inplace=True),
            nn.Linear(in_features=1024, out_features=1024),         # (FC2) b x 1024
            nn.Linear(in_features=1024, out_features=2)             # (FC3) b x 1024
        )
    
    def forward(self, X):
        X = self.featureNet(X)
        print(f"Size of Feature Net before reshape {X.size()}")
        X = X.view(-1, 256*4*4) # reshape for FC layers
        #X = torch.flatten(X)        # reshape for FC layers
        return self.regressor(X)

# testing script 
if __name__ == "__main__":
    test_tensor = (16, 3, 227, 227)
    model = SnoutNet()
    model = model.to(device)
    summary(model, input_size=test_tensor)