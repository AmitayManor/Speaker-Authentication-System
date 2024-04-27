import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter
from scipy.signal.windows import hamming
import os
import matplotlib.pyplot as plt

SAMPLE_RATE = 16000
ENERGY_THRESHOLD = 0.0001


def plot_audio(audio_data, title):

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

    starts = np.arange(0, len(audio_data) - win_length, step)
    ends = starts + win_length

    silent_mask = np.zeros_like(audio_data, dtype=bool)
    for start, is_silent in zip(starts, silent_indices):
        silent_mask[start:start + win_length] = is_silent

    non_silent_audio = audio_data[~silent_mask]

    return non_silent_audio


def pre_emphasis_filter(signal, alpha=0.97):
    return lfilter([1, -alpha], 1, signal)


def remove_dc(signal):
    return signal - np.mean(signal)


def energy_rate(signal, win_length, overlap, sample_rate):
    step = win_length - overlap
    energy = []
    frames = range(0, len(signal) - win_length, step)
    for i in frames:
        windowed_signal = signal[i:i + win_length] * hamming(win_length)
        energy.append(np.sum(windowed_signal ** 2) / float(win_length))

    energy = np.array(energy)

    normalized_energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy)) if np.max(energy) != np.min(
        energy) else np.zeros_like(energy)

    return normalized_energy, np.array(frames) / float(sample_rate)