import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import numpy as np


class CNNModel(nn.Module):
    def __init__(self, input_size):
        super(CNNModel, self).__init__()
        self.norm = nn.BatchNorm1d(input_size)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=11, kernel_size=2, padding='valid')
        self.bn1 = nn.BatchNorm1d(11)
        self.conv2 = nn.Conv1d(in_channels=11, out_channels=7, kernel_size=2, padding='valid')
        self.bn2 = nn.BatchNorm1d(7)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.4)
        
        self.fc1 = nn.Linear(7 * ((input_size - 3) // 2), 50)
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(50, 30)
        self.dropout3 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(30, 12)
        self.fc4 = nn.Linear(12, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Reshape to (batch, channels, features)
        x = self.norm(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout3(x)
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x.squeeze()
