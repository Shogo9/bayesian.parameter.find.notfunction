import optuna
from train import train_model  # モデルの訓練関数をインポート

def objective(trial):
    # Optunaのobjective関数として、train.pyで定義したtrain_model関数を使用
    return train_model(trial)

def perform_bayesian_optimization():
    # Optunaでハイパーパラメータ最適化のためのStudyを作成
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)  # 試行回数は任意

    # 最適な結果を表示
    print("Best Hyperparameters:", study.best_params)
    print("Best Loss:", study.best_value)

    # 学習済みのモデルパラメータを保存
    best_params = study.best_params
    return best_params

if __name__ == '__main__':
    best_params = perform_bayesian_optimization()
    print("Optimization completed. Best parameters:", best_params)
