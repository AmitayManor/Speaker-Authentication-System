from audio_recordings import record_audio
from features_extraction import compute_lpcc, lpc_to_cepstral
import numpy as np


def generate_sine_wave(frequency, sample_rate, duration):
    """ Generate a sine wave of a given frequency and duration. """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return 0.5 * np.sin(2 * np.pi * frequency * t)

def test_lpcc_with_sine_wave():
    """ Test LPCC calculation on a known signal (sine wave). """
    # Generate a 440 Hz sine wave
    sample_rate = 16000
    duration = 1  # 1 second
    signal = generate_sine_wave(440, sample_rate, duration)

    # Compute LPCC
    lpcc_features = compute_lpcc(signal, sr=sample_rate, order=12)

    # Assert the output is of the correct length and type
    assert len(lpcc_features) == 13, "LPCC feature length should be order + 1"
    assert isinstance(lpcc_features, np.ndarray), "Output should be a numpy array"
    print(f"lpcc_features with sine wave: {lpcc_features}")

def test_lpcc_with_zero_input():
    """ Test LPCC calculation with zero input. """
    sample_rate = 16000
    duration = 1  # 1 second
    signal = np.zeros(int(sample_rate * duration))

    # Compute LPCC
    lpcc_features = compute_lpcc(signal, sr=sample_rate, order=12)

    # Assert non-nan or inf values
    assert not np.any(np.isnan(lpcc_features)), "Should not contain NaN"
    assert not np.any(np.isinf(lpcc_features)), "Should not contain Inf"
    print(f"lpcc_features with zero input: {lpcc_features}")

def test_lpcc_with_random_noise():
    """ Test LPCC calculation with random noise. """
    np.random.seed(0)  # for reproducibility
    sample_rate = 16000
    duration = 1  # 1 second
    signal = np.random.randn(int(sample_rate * duration))

    # Compute LPCC
    lpcc_features = compute_lpcc(signal, sr=sample_rate, order=12)

    # Basic assertion on output size
    assert len(lpcc_features) == 13, "LPCC feature length should be order + 1"
    print(f"lpcc_features with random noise: {lpcc_features}")

if __name__ == "__main__":
    test_lpcc_with_sine_wave()
    test_lpcc_with_zero_input()
    test_lpcc_with_random_noise()
    print("All tests passed!")