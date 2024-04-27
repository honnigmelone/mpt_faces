import torch
from torch import nn

# NOTE: This will be the network architecture. 

class Net(nn.Module):
    def __init__(self, nClasses):
        super().__init__()


        # TODO: Implement module constructor. --> Innit ist constructor
        # Define network architecture as needed --> CNN
        # Input images will be 3 channels 256x256 pixels. --> !!
        # Output must be a nClasses Tensor. --> get count of folders, that will be the number of output classes
        pass
    def forward(self, x):
        # TODO: 
        # Implement forward pass
        #  x is a BATCH_SIZEx3x256x256 Tensor
        #  return value must be a BATCH_SIZExN_CLASSES Tensor
        pass
