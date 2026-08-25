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

# Custom initializer to handle extra arguments in deserialization
from keras.initializers import GlorotUniform

class CustomGlorotUniform(GlorotUniform):
    """Custom initializer that ignores unexpected 'input_axes' and 'output_axes' arguments."""
    def __init__(self, seed=None, input_axes=None, output_axes=None, **kwargs):
        super().__init__(seed=seed, **kwargs)


def load_model_and_scaler(model_path, scaler_path):
    """
    Load the trained Keras model (without compiling) and fitted scaler.
    Uses custom_objects to handle version differences in initializers.
    """
    # Register custom objects
    custom_objects = {'GlorotUniform': CustomGlorotUniform}
    
    # Load without compiling to bypass metric deserialization
    model = tf.keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
        compile=False
    )
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