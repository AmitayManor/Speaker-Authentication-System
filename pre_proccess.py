import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter
from scipy.signal.windows import hamming
import os
import matplotlib.pyplot as plt

SAMPLE_RATE = 16000
ENERGY_THRESHOLD = 0.0001

"""     Work FLow:      
        
        1. Reading .wav Audio files from test sub folder (about 10 speakers)
        2. Pre-process each .wav audio file:
            1. pre emphasize
            2. dc removal
            3. calculate energy (and ZCR if needed)
            4. classify silence segments in the file (by energy level)
            5. trim silence segments
        3. Returns processed data (format: {[audio data] : 'Speaker name'})
        4. Save the new data in another sub folder
        5. Moving to Features extractions
        
"""


def plot_audio(audio_data, title):
    """
    Plot the waveform of an audio signal.

    Parameters:
    - audio_data: numpy array of the audio samples.
    - title: title of the plot.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(audio_data, color='blue')
    plt.title(title)
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()


def process_audio_folder(folder_path):
    processed_data = {}
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".wav"):
                full_path = os.path.join(root, file)
                try:
                    data = process_audio(full_path)
                    processed_data[os.path.splitext(file)[0]] = data
                except Exception as e:
                    print(f"Error processing {file}: {e}")
    return processed_data


def process_audio(file_path, win_length=1024, overlap=512):
    rate, audio = wavfile.read(file_path)
    audio = pre_emphasis_filter(audio)
    audio = remove_dc(audio)
    energy, _ = energy_rate(audio, win_length, overlap, rate)
    audio = remove_silence(audio, energy, win_length, overlap)
    return audio


def remove_silence(audio_data, energy, win_length, overlap, threshold=ENERGY_THRESHOLD):
    step = win_length - overlap
    silent_indices = energy <= threshold

    # Create an array of indices that correspond to the start of each window
    starts = np.arange(0, len(audio_data) - win_length, step)
    ends = starts + win_length

    # Create a mask for all indices that are silent
    silent_mask = np.zeros_like(audio_data, dtype=bool)
    for start, is_silent in zip(starts, silent_indices):
        silent_mask[start:start + win_length] = is_silent

    # Use the inverse of the mask to select non-silent segments
    non_silent_audio = audio_data[~silent_mask]

    return non_silent_audio


def pre_emphasis_filter(signal, alpha=0.97):
    return lfilter([1, -alpha], 1, signal)


# DC removal
def remove_dc(signal):
    return signal - np.mean(signal)


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