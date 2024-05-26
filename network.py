from torch import nn


# NOTE: This is the network architecture.

class Net(nn.Module):
    def __init__(self, nClasses):
        super().__init__()

        # TODO: Implement module constructor.               --> Innit ist constructor
        # Define network architecture as needed             --> CNN
        # Input images will be 3 channels 256x256 pixels.   --> Input-Channel = 3
        # Output must be a nClasses Tensor.                 --> last linear layer has to have nClasses output

        # Convolutional layers:
        self.Conv1 = nn.Conv2d(3, 15, (3, 3), padding='same')
        self.Conv2 = nn.Conv2d(15, 45, (3, 3), padding='same')
        self.Conv3 = nn.Conv2d(45, 10, (3, 3), padding='same')

        # Other Layers:
        self.Pool = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.Flat = nn.Flatten()
        self.Relu = nn.ReLU()
        self.Soft = nn.Softmax(dim=1)

        # Input is calculated with size*size*output of last layer.
        # 64x64x10 = 40960
        self.fc = nn.Linear(40960, nClasses)

    def forward(self, x):
        # TODO:
        # Implement forward pass
        #  x is a BATCH_SIZEx3x256x256 Tensor
        #  return value must be a BATCH_SIZExN_CLASSES Tensor

        x = self.Conv1(x)   # Input=256x256x3;   Outbut=256x256x15
        x = self.Relu(x)    # Input=256x256x15;  Outbut=256x256x15
        x = self.Conv2(x)   # Input=256x256x15;  Outbut=256x256x45
        x = self.Pool(x)    # Input=256x256x45;  Outbut=128x128x45
        x = self.Conv3(x)   # Input=128x128x45;  Outbut=128x128x10
        x = self.Pool(x)    # Input=128x128x10;  Outbut=64x64x10

        x = self.Flat(x)    # Input=64x64x10;    Outbut=64x64x10
        x = self.Relu(x)    # Input=64x64x10;    Outbut=64x64x10

        x = self.fc(x)      # Input=64x64x10;    Outbut=64x64x10
        x = self.Soft(x)    # Input=64x64x10;    Outbut=64x64x10
        return x
