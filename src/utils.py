import json
import torch
from torch.utils.data import Dataset, DataLoader

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

def get_dataloader(json_path, batch_size=16):
    dataset = CustomDataset(json_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
