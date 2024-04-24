import numpy as np
import librosa
import matplotlib.pyplot as plt
from spafe.features.lpc import lpc2lpcc
from spafe.utils.vis import show_features

"""     Work Flow:

        1. Read processed audio data from relevant sub folder
        2. Compute LPCC:
            1. Compute LPC
            2. Compute LPC to Cepstral
        3. Save new data in a new sub folder (format: {[data] : 'Speaker name'})
        4. Move to next step: classify with Gauss NB

"""


def plot_lpcc_features(speaker_features):
    """
    Plot LPCC features for each speaker in the dataset.

    Args:
        speaker_features (dict): A dictionary mapping each speaker to their LPCC features.
    """
    num_speakers = len(speaker_features)
    fig, axs = plt.subplots(num_speakers, figsize=(15, num_speakers * 2), sharex=True)

    # If there's only one speaker, axs is not a list, so we wrap it in a list.
    if num_speakers == 1:
        axs = [axs]

    # Go through each speaker and plot their LPCC features
    for ax, (speaker, features) in zip(axs, speaker_features.items()):
        ax.plot(features, label=f'Speaker: {speaker}')
        ax.set_title(f'LPCC Features for {speaker}')
        ax.legend()
        ax.grid(True)

    plt.xlabel('LPCC Coefficient Index')
    plt.ylabel('Coefficient Value')
    fig.tight_layout()
    plt.show()


def extract_features_from_processed_data(processed_data, order=20):
    """
    Takes pre-processed audio data and computes LPCC features for each speaker.

    Args:
        processed_data (dict): Dictionary where keys are speaker names and values are their audio data.
        sr (int): The sample rate of the audio signals.
        order (int): The order of the linear prediction.

    Returns:
        dict: A dictionary mapping each speaker to their LPCC features.
    """
    speaker_features = {}
    for speaker, signal in processed_data.items():
        lpcc_features = compute_lpcc(signal, order)
        speaker_features[speaker] = lpcc_features

    return speaker_features


def extract_features_from_processed_data_test(processed_data, order=20):
    """
    Takes pre-processed audio data and computes LPCC features for each speaker.

    Args:
        processed_data (dict): Dictionary where keys are speaker names and values are their audio data.
        sr (int): The sample rate of the audio signals.
        order (int): The order of the linear prediction.

    Returns:
        dict: A dictionary mapping each speaker to their LPCC features.
    """
    speaker_features = {}
    for speaker, signal in processed_data.items():
        lpcc_features = compute_lpcc_test(signal, order)
        speaker_features[speaker] = lpcc_features

    return speaker_features


def compute_lpcc(signal, order):
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
    lpc_test = lpc2lpcc(lpc_coeffs, order, order)

    return lpc_test


def compute_lpcc_test(signal, order):
    lpc_coeffs = librosa.lpc(signal, order=order)
    lpc_test = lpc_to_cepstral(lpc_coeffs, order)
    return lpc_test

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
