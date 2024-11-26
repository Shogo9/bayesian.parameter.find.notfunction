import torch
from model import SimpleModel


# 推論用関数
def generate_output(input_data, model_path='../results/best_model.pth'):
    """入力データに基づいて推論を生成"""
    model = SimpleModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    input_scaler = joblib.load('../data/input_scaler.pkl')
    output_scaler = joblib.load('../data/output_scaler.pkl')

    input_data_scaled = input_scaler.transform([input_data])
    input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32)

    with torch.no_grad():
        output_scaled = model(input_tensor)
        output = output_scaler.inverse_transform(output_scaled.numpy())

    return output[0]

if __name__ == '__main__':
    # 推論テスト
    sample_input = [73, 22, 22]
    output = generate_output(sample_input)
    print(f"Generated Output: {output}")