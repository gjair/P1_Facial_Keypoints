import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=0)

        # Max-Pool layer that we will use multiple times
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
#         # Dropout layers
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.3)
        self.dropout4 = nn.Dropout(p=0.4)
        self.dropout5 = nn.Dropout(p=0.5)
        self.dropout6 = nn.Dropout(p=0.6)
        
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=10*10*256, out_features=1024)
#         self.fc2 = nn.Linear(in_features=1024, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=136)
        
       
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        ## Conv layers
        x = self.pool(F.relu(self.conv1(x)))
#         x = self.dropout1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout3(x)
        x = self.pool(F.relu(self.conv3(x)))
#         x = self.dropout3(x)
        x = self.pool(F.relu(self.conv4(x)))
#         x = self.dropout4(x)
        ## Flatten
        x = x.view(x.size(0), -1) # .view() can be thought as np.reshape
        ## Fully connected layers
        x = F.relu(self.fc1(x))
#         x = self.dropout5(x)
#         x = F.relu(self.fc2(x))
#         x = self.dropout6(x)
        x = self.fc3(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x

        