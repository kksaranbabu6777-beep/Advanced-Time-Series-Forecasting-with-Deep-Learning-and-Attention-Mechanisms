import numpy as np
import pandas as pd

def generate_synthetic_data(n_obs=5000, n_features=5):
    np.random.seed(42)
    t = np.arange(n_obs)
    data = {}
    for i in range(n_features):
        trend = 0.01 * t
        seasonal = 5 * np.sin(2 * np.pi * t / (365 - 10 * i))
        noise = np.random.normal(0, 1.5, n_obs)
        data[f'feat_{i}'] = trend + seasonal + noise
    df = pd.DataFrame(data)
    df['target'] = 0.2*df['feat_0'] + 0.5*df['feat_1'] + np.roll(df['feat_2'], 1) + np.random.normal(0,1,n_obs)
    df.to_csv('data/synthetic_multivariate_timeseries.csv', index=False)
    return df

if _name_ == '_main_':
    generate_synthetic_data()
