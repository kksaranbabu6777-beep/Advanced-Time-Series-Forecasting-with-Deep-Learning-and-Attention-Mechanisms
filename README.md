# Advanced Time Series Forecasting with Deep Learning and Attention Mechanisms

A project exploring state-of-the-art time series forecasting using LSTM, VAR, and Transformer Encoder attention models. Includes fully-reproducible code, orchestrated experiments, and interpretability analysis.

## Structure

- data/: Synthetic/generated datasets
- models/: Core PyTorch models (LSTM, VAR, Transformer)
- scripts/: End-to-end training, evaluation, tuning
- notebooks/: Interactive analysis and visualizations
- utils/: Data wrangling and pipeline helpers

## Quick Start

1. pip install -r requirements.txt
2. Generate dataset: python utils/data_pipeline.py
3. Train models: python scripts/train.py --model lstm or --model transformer
4. Hyperparameter tuning: python scripts/hyperparam_opt.py
5. Review results/notebooks

Tested in Python 3.9+
