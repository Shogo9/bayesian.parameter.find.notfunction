import torch
from model import SimpleModel

def generate_output(input_data, model_path='results/best_model.pth'):
    model = SimpleModel()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    # 入力データをテンソルに変換し、生成
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor)
    return output.numpy()

# 生成AIテスト
if __name__ == '__main__':
    sample_input = [0.15, -0.1, 0.05]  # 例: 任意の3次元入力
    output = generate_output(sample_input)
    print(f"Generated Output: {output}")