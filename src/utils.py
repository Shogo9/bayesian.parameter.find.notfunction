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

# CPUスレッド数を設定
torch.set_num_threads(8)

# JSONデータの正規化と保存
def normalize_json_data(input_path, output_path):
    """データを正規化して保存"""
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {input_path} not found.")
        return

    inputs = np.array(data["inputs"])
    outputs = np.array(data["outputs"])

    # スケーラーのインスタンス化
    input_scaler = MinMaxScaler()
    output_scaler = MinMaxScaler()

    # 入力と出力のスケーリング
    inputs_normalized = input_scaler.fit_transform(inputs)
    outputs_normalized = output_scaler.fit_transform(outputs)

    # 正規化されたデータを保存
    normalized_data = {
        "inputs": inputs_normalized.tolist(),
        "outputs": outputs_normalized.tolist()
    }

    try:
        with open(output_path, 'w') as f:
            json.dump(normalized_data, f)
    except IOError:
        print(f"Error: Unable to write to {output_path}.")
        return

    # スケーラーを保存
    joblib.dump(input_scaler, '../data/input_scaler.pkl')
    joblib.dump(output_scaler, '../data/output_scaler.pkl')
    print("Data normalization completed and scalers saved.")
    
    # カスタムデータセットクラス
class CustomDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.inputs = data["inputs"]
        self.outputs = data["outputs"]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)
        y = torch.tensor(self.outputs[idx], dtype=torch.float32)
        return x, y

# データローダーの生成
def get_dataloader(json_path, batch_size=16):
    dataset = CustomDataset(json_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
