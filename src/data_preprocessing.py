"""
src/data_preprocessing.py
Data loading, cleaning, normalisation, and sequence creation for LSTM.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os


def load_data(filepath):
    """
    Load CSV with a 'timestamp' column and set it as index.
    """
    df = pd.read_csv(filepath, parse_dates=['timestamp'], index_col='timestamp')
    return df


def clean_data(df, feature_columns):
    """
    Select features, forward/backward fill missing values.
    """
    data = df[feature_columns].copy()
    # Fill missing values
    data = data.ffill().bfill()
    return data


def get_scaler(data, fit=True, scaler_path=None):
    """
    Fit a MinMaxScaler on the data or load an existing one.
    If fit=True, fit and optionally save; else load from scaler_path.
    """
    if fit:
        scaler = MinMaxScaler()
        scaler.fit(data)
        if scaler_path:
            joblib.dump(scaler, scaler_path)
        return scaler
    else:
        if scaler_path is None:
            raise ValueError("scaler_path must be provided when fit=False")
        return joblib.load(scaler_path)


def scale_data(data, scaler, fit=False):
    """
    Scale data using the provided scaler. If fit=True, fit the scaler first.
    """
    if fit:
        scaled = scaler.fit_transform(data)
    else:
        scaled = scaler.transform(data)
    return scaled


def create_sequences(data, timesteps):
    """
    Convert 2D array (samples, features) into 3D sequences (samples, timesteps, features).
    """
    X = []
    for i in range(timesteps, len(data)):
        X.append(data[i - timesteps:i])
    return np.array(X)


def get_preprocessed_data(filepath, feature_columns, timesteps, scaler_path=None, fit_scaler=True):
    """
    High-level function: load, clean, scale, and create sequences.
    Returns: X (sequences), scaler (fitted or loaded)
    """
    # 1. Load
    df = load_data(filepath)
    # 2. Clean
    data_clean = clean_data(df, feature_columns)
    # 3. Scale
    scaler = get_scaler(data_clean, fit=fit_scaler, scaler_path=scaler_path)
    scaled = scale_data(data_clean, scaler, fit=fit_scaler)
    # 4. Create sequences
    X = create_sequences(scaled, timesteps)
    return X, scaler