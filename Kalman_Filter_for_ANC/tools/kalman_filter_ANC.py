import numpy as np
import torch

def kalman_step(w_prev, P_prev, x, d, q):
    """
    * Single step update of the Kalman filter algorithm.
    @ Parameters:
    - w_prev  - State estimate from the previous time step (numpy array)
    - P_prev  - State error covariance matrix from the previous time step (numpy array)
    - x       - Current input data (numpy array)
    - d       - Current measurement data (scalar or numpy array)
    - q       - Measurement noise covariance (scalar)
    @ Returns:
    - w_hat      - Predicted new state (numpy array)
    - P          - Predicted state error covariance matrix (numpy array)
    - K          - Kalman gain (numpy array)
    - w          - Updated state estimate (numpy array)
    - P_updated  - Updated state error covariance matrix (numpy array)
    """
    # Predict the new state
    w_hat = w_prev
    # Predict the state error covariance matrix
    P = P_prev + q
    # Compute the Kalman gain
    K = P @ x / (x.t() @ P @ x + q)
    # Estimate the state
    if isinstance(d, (int, float)):
        d_scalar = d
    else:
        d_scalar = d.item()  # Convert numpy scalar to Python scalar if d is a numpy array
    w = w_hat + K * (d_scalar - x.t() @ w_hat)
    # Update the state error covariance matrix
    P_updated = (torch.eye(len(K)) - K @ x.t()) @ P
    return w_hat, P, K, w, P_updated