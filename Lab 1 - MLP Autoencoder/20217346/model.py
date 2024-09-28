"""
Implements forward(X) == decode(encode(X))
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

class autoencoderMLP4Layer(nn.Module):
    def __init__(self, N_input: int = 784, N_bottleneck: int = 8, N_output:int=784):
        super(autoencoderMLP4Layer, self).__init__()    # call to nn.Module init()
        N2 = 392
        IMG_SIZE = 784
        
        # create the 4 (linear) layers 
        self.fc1 = nn.Linear(N_input, N2)
        self.fc2 = nn.Linear(N2, N_bottleneck)
        self.fc3 = nn.Linear(N_bottleneck, N2)
        self.fc4 = nn.Linear(N2, N_output)
        self.type = 'MLP4'
        self.input_shape = (1, IMG_SIZE)

    # in general --> model.forward() = model.decode(model.encode())
    def forward(self, X: torch.Tensor)->torch.Tensor:
        return self.decode(self.encode(X))    

    # the tensor returned is the BOTTLENECK TENSOR (size = N_bottleneck)
    # X in this case would be the raw 28x28 image
    def encode(self, X: torch.Tensor)->torch.Tensor:
        X = self.fc1(X)
        X = F.relu(X)
        X = self.fc2(X)
        X = F.relu(X)
        return X
    
    # tensor input will be a bottleneck
    # tensor output will be a generated image in flattened form (1, 784) that needs to be shaped to (28, 28) for matplotlib
    def decode(self, X: torch.Tensor)->torch.Tensor:
        X = self.fc3(X)
        X = F.relu(X)
        X = self.fc4(X)
        X = torch.sigmoid(X)
        return X
    
        