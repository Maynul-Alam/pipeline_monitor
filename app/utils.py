"""
app/utils.py
Utilities for the Streamlit app: load model/scaler, preprocess new data.
"""
import sys
import os

# Ensure the project root is in sys.path so we can import 'src' modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from src.data_preprocessing import clean_data, scale_data, create_sequences

# ------------------------------------------------------------------
# MONKEY-PATCH: Force GlorotUniform to accept extra arguments
# This fixes the "unexpected keyword argument 'input_axes'" error.
# ------------------------------------------------------------------
from keras.initializers import GlorotUniform

_original_glorot_init = GlorotUniform.__init__

def _patched_glorot_init(self, seed=None, input_axes=None, output_axes=None, **kwargs):
    # Call the original __init__ without the extra arguments
    _original_glorot_init(self, seed=seed, **kwargs)

GlorotUniform.__init__ = _patched_glorot_init

# ------------------------------------------------------------------

def load_model_and_scaler(model_path, scaler_path):
    """
    Load the trained Keras model (without compiling) and fitted scaler.
    The monkey‑patch above ensures initializers load correctly.
    """
    # Load without compiling to bypass metric deserialization
    model = tf.keras.models.load_model(model_path, compile=False)
    # Recompile with the same loss as during training
    model.compile(optimizer='adam', loss='mse')
    scaler = joblib.load(scaler_path)
    return model, scaler


def preprocess_new_data(df, feature_columns, timesteps, scaler):
    """
    Clean, scale, and sequence new incoming data (for prediction).
    Returns: sequences ready for the model.
    """
    data_clean = clean_data(df, feature_columns)
    scaled = scale_data(data_clean, scaler, fit=False)
    X = create_sequences(scaled, timesteps)
    return X


def get_reconstruction_error(model, X):
    """Compute MSE per sequence."""
    reconstructions = model.predict(X, verbose=0)
    mse = np.mean(np.power(X - reconstructions, 2), axis=(1, 2))
    return mse