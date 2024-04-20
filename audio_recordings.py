import sounddevice as sd

def record_audio(duration=5, sample_rate=16000):
    """
    Records audio from the microphone.
    :param duration: Length of the recording in seconds.
    :param sample_rate: Sampling rate of the audio in Hz.
    :return: Numpy array of recorded audio.
    """
    print("Recording...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float64')
    sd.wait()  # Wait until recording is finished
    print("Recording complete")
    return recording.flatten()