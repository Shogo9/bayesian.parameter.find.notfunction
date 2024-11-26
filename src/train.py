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

def train_model(trial):
    """Optunaによるハイパーパラメータ最適化のためのトレーニング"""
    # ベイズ最適化でハイパーパラメータを探索
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    batch_size = trial.suggest_int('batch_size', 8, 64)

    model = SimpleModel()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataloader = get_dataloader('../data/data_normalized.json', batch_size=batch_size)

    model.train()
    best_loss = float('inf')
    best_model = None

    num_epochs = 10
    progress_bar = tqdm(range(num_epochs), desc="Training Progress", unit="epoch")
    for epoch in progress_bar:
        epoch_loss = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model = model.state_dict()

        progress_bar.set_description(f"Epoch {epoch+1} | Loss: {avg_loss:.6f}")

    if not os.path.exists('../results'):
        os.makedirs('../results')
    torch.save(best_model, '../results/best_model.pth')

    return best_loss