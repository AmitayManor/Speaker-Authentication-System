import numpy as np
import librosa

def compute_lpcc(signal, sr, order):
    """
    Compute Linear Prediction Cepstral Coefficients (LPCC) from an audio signal.

    Args:
        signal (array): The audio signal from which to compute features.
        sr (int): The sample rate of the audio signal.
        order (int): The order of the linear prediction.

    Returns:
        np.array: The computed LPCC features.
    """
    # Step 1: Compute the LPC coefficients
    lpc_coeffs = librosa.lpc(signal, order=order)

    # Step 2: Convert LPC to LPCC
    lpcc = lpc_to_cepstral(lpc_coeffs, order)

    return lpcc

def lpc_to_cepstral(lpc_coeffs, order):
    """
    Convert LPC coefficients to Cepstral coefficients.

    Args:
        lpc_coeffs (array): LPC coefficients.
        order (int): The order of the LPC coefficients.

    Returns:
        np.array: Cepstral coefficients derived from LPC.
    """
    # Initialize cepstral coefficients array
    lpcc = np.zeros(order + 1)
    lpcc[0] = np.log(np.square(lpc_coeffs[0]))

    # Recursively compute cepstral coefficients
    for n in range(1, order + 1):
        sum_nk = sum((k / n) * lpcc[k] * lpc_coeffs[n - k] for k in range(1, n))
        lpcc[n] = -lpc_coeffs[n] + sum_nk

    return lpcc
