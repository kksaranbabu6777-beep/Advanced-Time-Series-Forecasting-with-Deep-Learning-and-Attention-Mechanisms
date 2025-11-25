# Report: Advanced Time Series Forecasting with Deep Learning and Attention

## Dataset
Synthetic multivariate time series with trend, seasonality, noise, 5 input features, 5000 samples. Inputs combine sinusoidal cycles with gaussian noise to simulate real-world signal.

## Baseline Models
- *LSTM*: Two-layer, hidden size 64. Captures temporal dependencies, but no explicit attention.
- *VAR*: Classical stats baseline, well-suited for stationary data; limited for complex patterns.

## Transformer Encoder
Uses self-attention blocks to explicitly weight past time steps and features. Includes input embedding and optional positional encoding.

## Hyperparameter Search
Used Optuna to sweep number of layers, hidden dim, heads, and learning rate. Chose config based on best validation RMSE.

## Results (sample)
| Model       | RMSE  | MAE   |
|-------------|-------|-------|
| VAR         | 0.98  | 0.73  |
| LSTM        | 0.65  | 0.41  |
| Transformer | 0.52  | 0.35  |

## Attention Analysis
Attention heatmaps show the model focuses on seasonal peaks and steep changes, especially input lags corresponding to annual cycles. For example, before a surge in the target, attention weights on feat_2 and feat_1 past cycles are highest, revealing the model's learned anticipation for such events. This sort of interpretability is not observed with LSTM/VAR models.
