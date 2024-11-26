import librosa
from spafe.features.lpc import lpc2lpcc


def extract_features_from_processed_data(processed_data, order=20):

    speaker_features = {}
    for speaker, signal in processed_data.items():
        lpcc_features = compute_lpcc(signal, order)
        speaker_features[speaker] = lpcc_features
    return speaker_features


def compute_lpcc(signal, order):

    lpc_coeffs = librosa.lpc(signal, order=order)
    lpc_test = lpc2lpcc(lpc_coeffs, order, order)
    return lpc_test
