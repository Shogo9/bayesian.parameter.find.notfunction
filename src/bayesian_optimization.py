import optuna
from train import train_model  # モデルの訓練関数をインポート

# Optunaのベイズ最適化
def perform_bayesian_optimization():
    """Optunaによるベイズ最適化の実行"""
    study = optuna.create_study(direction='minimize')
    n_trials = 10
    progress_bar = tqdm(total=n_trials, desc="Bayesian Optimization Progress", unit="trial")

    def callback(study, trial):
        progress_bar.set_description(f"Trial {trial.number+1} | Best Loss: {study.best_value:.6f}")
        progress_bar.update(1)

    study.optimize(train_model, n_trials=n_trials, callbacks=[callback])

    progress_bar.close()
    print("Best Hyperparameters:", study.best_params)
    print("Best Loss:", study.best_value)

    return study.best_params

if __name__ == '__main__':
    # JSONデータの正規化
    normalize_json_data('../data/data.json', '../data/data_normalized.json')

    # Optunaによるベイズ最適化の実行
    best_params = perform_bayesian_optimization()
    print("Optimization completed. Best parameters:", best_params)