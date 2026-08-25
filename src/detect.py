"""
src/detect.py
Reconstruction error computation, thresholding, and plotting.
"""
import numpy as np
import matplotlib.pyplot as plt
import os


def compute_reconstruction_error(model, X_data):
    """
    Reconstruct the input sequences and compute mean squared error per sample.
    Returns: 1D array of MSE values (one per sequence).
    """
    reconstructions = model.predict(X_data, verbose=0)
    mse = np.mean(np.power(X_data - reconstructions, 2), axis=(1, 2))
    return mse


def get_threshold(errors, multiplier=3):
    """
    Dynamic threshold = mean + multiplier * standard deviation.
    """
    threshold = np.mean(errors) + multiplier * np.std(errors)
    return threshold


def detect_anomalies(errors, threshold):
    """
    Return boolean array: True where error > threshold.
    """
    return errors > threshold


def plot_anomalies(errors, threshold, anomalies, save_path=None):
    """
    Plot reconstruction errors with threshold and marked anomalies.
    If save_path is given, save the figure; otherwise display it.
    """
    plt.figure(figsize=(14, 6))
    plt.plot(errors, label='Reconstruction Error', color='blue', alpha=0.7)
    plt.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold = {threshold:.4f}')
    
    # Mark anomalies
    anomaly_indices = np.where(anomalies)[0]
    if len(anomaly_indices) > 0:
        plt.scatter(anomaly_indices, errors[anomalies], color='red', s=30, label=f'Anomalies ({len(anomaly_indices)})')
    
    plt.xlabel('Sample Index')
    plt.ylabel('Reconstruction Error (MSE)')
    plt.title('Anomaly Detection in Pipeline Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def run_detection_pipeline(model, X_data, threshold_multiplier=3, plot_path=None):
    """
    High-level: compute errors, threshold, detect anomalies, and optionally plot.
    Returns: errors, threshold, anomalies
    """
    errors = compute_reconstruction_error(model, X_data)
    threshold = get_threshold(errors, multiplier=threshold_multiplier)
    anomalies = detect_anomalies(errors, threshold)
    
    if plot_path is not None:
        plot_anomalies(errors, threshold, anomalies, save_path=plot_path)
    
    return errors, threshold, anomalies