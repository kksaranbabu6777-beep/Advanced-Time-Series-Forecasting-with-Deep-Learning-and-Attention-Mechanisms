import optuna
import torch
from models.transformer_attention import TimeSeriesTransformer
# (Assume you have data prepared and a train/eval routine)

def objective(trial):
    d_model = trial.suggest_int('d_model', 16, 128, step=16)
    nhead = trial.suggest_categorical('nhead', [2, 4, 8])
    num_layers = trial.suggest_int('num_layers', 1, 4)
    lr = trial.suggest_loguniform("lr", 1e-4, 5e-3)
    # (Model, trainer, and evaluation logic here, returning validation RMSE)
    val_score = ... # replace with your val RMSE
    return val_score

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)
print("Best params:", study.best_params)
