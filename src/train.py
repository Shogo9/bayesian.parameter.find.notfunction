import torch
import torch.optim as optim
import optuna
from model import SimpleModel
from utils import get_dataloader
import torch.nn as nn
import os

def train_model(trial):
    # ベイズ最適化でハイパーパラメータを探索
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    batch_size = trial.suggest_int('batch_size', 8, 64)
    
    model = SimpleModel()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataloader = get_dataloader('data/data.json', batch_size=batch_size)

    model.train()
    best_loss = float('inf')  # 最良の損失値を保持する変数
    best_model = None  # 最良モデルを保持する変数

    for epoch in range(10):  # エポック数は任意
        epoch_loss = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # 各エポック終了後に損失を計算し、最良のモデルを保存
        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch {epoch+1}, Loss: {avg_loss}')

        if avg_loss < best_loss:  # 最良の損失を更新
            best_loss = avg_loss
            best_model = model.state_dict()  # 最良モデルのパラメータを保存

    # 最良のモデルを保存
    if not os.path.exists('results'):
        os.makedirs('results')
    torch.save(best_model, 'results/best_model.pth')
    print(f'Best model saved to results/best_model.pth')

    return best_loss  # 最良損失を返す

# Optunaのベイズ最適化
if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    study.optimize(train_model, n_trials=50)
    print(f"Best params: {study.best_params}")
    print(f"Best loss: {study.best_value}")
