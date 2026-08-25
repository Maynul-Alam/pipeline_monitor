"""
src/train.py
Training the LSTM Autoencoder and saving artifacts.
"""
import tensorflow as tf
from src.model import build_model
import joblib
import os


def train_model(X_train, X_val, timesteps, n_features, lstm_units=64, epochs=50, batch_size=64):
    """
    Build and train the autoencoder.
    Returns: trained model, training history.
    """
    model = build_model(timesteps, n_features, lstm_units)
    history = model.fit(
        X_train, X_train,                # autoencoder: target = input
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, X_val),
        shuffle=False,                   # never shuffle time series
        verbose=1
    )
    return model, history


def save_model(model, model_path, scaler, scaler_path):
    """
    Save the trained model (using SavedModel format for better compatibility)
    and the fitted scaler.
    """
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    
    # Save model in SavedModel format (more robust across versions)
    model.save(model_path, save_format='tf')
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path} (SavedModel format)")
    print(f"Scaler saved to {scaler_path}")