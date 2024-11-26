import torch
import json
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import optuna
from torchsummary import summary
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib  # スケーラーの保存用
import sys
import io
from torchviz import make_dot

# モデル定義
class SimpleModel(nn.Module):
    def __init__(self, input_dim=3, output_dim=2):
        super(SimpleModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.network(x)