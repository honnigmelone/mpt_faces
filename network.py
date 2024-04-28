import torch
from torch import nn

# NOTE: This will be the network architecture. 

class Net(nn.Module):
    def __init__(self, nClasses):
        super().__init__()

        # TODO: Implement module constructor.               --> Innit ist constructor
        # Define network architecture as needed             --> CNN
        # Input images will be 3 channels 256x256 pixels.   --> Input-Channel = 3
        # Output must be a nClasses Tensor.                 --> last linear layer has to have nClasses output


        # Convolutional and Maxpooling layers:
        self.Conv1 = nn.Conv2d(3,16,(3,3)) # default padding=same
        self.Conv2 = nn.Conv2d(16,32,(3,3))
        self.Conv3 = nn.Conv2d(32,64,(3,3))

        self.Conv4 = nn.Conv2d(64,32,(3,3))
        self.Conv5 = nn.Conv2d(32,16,(3,3))

        # Input is calculated with size*size*output of last layer. Size here depends on all the strides of maxPool layers before
        # Here I'm planning only one with stride 2 --> 256/2 = 128
        self.fc1 = nn.Linear((128*128*16), 512) 
        self.fc2 = nn.Linear(512, nClasses)

        # Other Layers:
        self.Pool = nn.MaxPool2d((2,2), stride=(2,2))
        self.Relu = nn.ReLU()
        self.Flat = nn.Flatten()
        self.Soft = nn.Softmax(dim=1) # adjust dim maybe?
        




    def forward(self, x):
        # TODO: 
        # Implement forward pass
        #  x is a BATCH_SIZEx3x256x256 Tensor
        #  return value must be a BATCH_SIZExN_CLASSES Tensor
        pass
