import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter
from scipy.signal.windows import hamming
import pandas as pd

# Define your classification thresholds
ENERGY_THRESHOLD = 0.001
def remove_silence(audio_data, threshold=ENERGY_THRESHOLD):
    """Removes silence from an audio signal. threshold sets by the """
    # Assuming audio_data is a 1-D numpy array
    return audio_data[np.abs(audio_data) > threshold]

# Function to classify each frame based on a tuple of ZCR and energy
def classify_frame(energy):
    # Define clear, non-overlapping conditions for each classification
    if energy < ENERGY_THRESHOLD:
        return 'silence'  # Low ZCR and low energy indicate silence


# Main processing function
def process_audio(pcm_path, sample_rate=16000, win_length=1024, overlap=512):
    audio = read_pcm_file(pcm_path, sample_rate)
    audio = pre_emphasis_filter(audio)
    audio = remove_dc(audio)
    zcr, zcr_time = zero_crossing_rate(audio, win_length, overlap, sample_rate)
    energy, energy_time = energy_rate(audio, win_length, overlap, sample_rate)

    data_output = list(zip(zcr, energy))
    classifications = [classify_frame(zcr, energy) for zcr, energy in data_output]

    return audio, zcr, zcr_time, energy, energy_time, data_output, classifications


# Load the PCM file
def read_pcm_file(file_path, sample_rate=16000):
    with open(file_path, 'rb') as pcm_file:
        pcm_data = np.fromfile(pcm_file, dtype=np.int16)
    # Write to WAV file
    wavfile.write('phn_saf1.wav', sample_rate, pcm_data)
    return pcm_data


# Pre-emphasis filter
def pre_emphasis_filter(signal, alpha=0.97):
    return lfilter([1, -alpha], 1, signal)


# DC removal
def remove_dc(signal):
    return signal - np.mean(signal)


# Zero-Crossing Rate calculation
def zero_crossing_rate(signal, win_length, overlap, sample_rate):
    step = win_length - overlap
    zcr = []
    frames = range(0, len(signal) - win_length, step)
    for i in frames:
        windowed_signal = signal[i:i + win_length] * hamming(win_length)
        crossings = np.where(np.diff(np.sign(windowed_signal)))[0]
        zcr.append((len(crossings)) / float(win_length))
    return np.array(zcr), np.array(frames) / float(sample_rate)


# Energy Rate calculation
def energy_rate(signal, win_length, overlap, sample_rate):
    step = win_length - overlap
    energy = []
    frames = range(0, len(signal) - win_length, step)
    for i in frames:
        windowed_signal = signal[i:i + win_length] * hamming(win_length)
        energy.append(np.sum(windowed_signal ** 2) / float(win_length))
        # Convert to numpy array for easier manipulation
    energy = np.array(energy)

    # Normalize the energy values
    normalized_energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy)) if np.max(energy) != np.min(
        energy) else np.zeros_like(energy)

    return normalized_energy, np.array(frames) / float(sample_rate)
