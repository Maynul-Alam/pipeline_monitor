"""
src/model.py
LSTM Autoencoder model definition.
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense


def build_model(timesteps, n_features, lstm_units=64):
    """
    Build and compile the LSTM Autoencoder.
    Returns: compiled Keras model.
    """
    model = tf.keras.Sequential([
        # Encoder
        LSTM(lstm_units, activation='relu', input_shape=(timesteps, n_features), return_sequences=False),
        # Bottleneck - RepeatVector to match timesteps for decoder
        RepeatVector(timesteps),
        # Decoder
        LSTM(lstm_units, activation='relu', return_sequences=True),
        # Output layer - reconstructs all features at each timestep
        TimeDistributed(Dense(n_features))
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model