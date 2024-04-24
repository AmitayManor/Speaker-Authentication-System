import librosa
import numpy as np
import os
import sqlite3
import pickle
from features_extraction import compute_lpcc
from sklearn.naive_bayes import GaussianNB
import json


def load_audio_file(file_path, sr=16000):
    """Load an audio file at the given sample rate."""
    audio, sr = librosa.load(file_path, sr=sr)
    return audio


def process_tedlium_dataset(directory_path):
    """Process all audio files in the specified directory of the TED-LIUM dataset."""
    # Maybe it should be the train subfolder of the various speakers

    lpcc_features = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(directory_path, filename)
            audio = load_audio_file(file_path)
            #TODO: Implement pre-proccessing functions on the data base
            lpcc = compute_lpcc(audio, order=12)
            lpcc_features.append(lpcc)

    return lpcc_features

def save_features(features, filename="features.npy"):
    """Save the computed features to a file."""
    np.save(filename, np.array(features))


def create_database(db_path='features.db'):
    """ Create a SQLite database to store audio features and speaker labels. """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS features (
            id INTEGER PRIMARY KEY,
            speaker_id TEXT NOT NULL,
            features TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()


def insert_features(db_path, speaker_id, features):
    """ Insert feature vectors and corresponding speaker IDs into the database. """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Convert list of features to a comma-separated string for storage
    features_str = ','.join(map(str, features))
    cursor.execute('INSERT INTO features (speaker_id, features) VALUES (?, ?)', (speaker_id, features_str))
    conn.commit()
    conn.close()


def load_features(db_path):
    """ Load all features from the database and return as NumPy arrays. """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT speaker_id, features FROM features')
    data = cursor.fetchall()
    conn.close()
    X = []
    y = []
    for speaker_id, feature_str in data:
        features = np.array(list(map(float, feature_str.split(','))))
        X.append(features)
        y.append(speaker_id)
    return np.array(X), np.array(y)


def train_classifier(X, y):
    """ Train a Gaussian Naive Bayes classifier.
    x = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    y = np.array([1, 1, 1, 0, 0, 0])"""

    clf_pf = GaussianNB(priors=[0.1, 0.9])
    clf_pf.fit(X, y)
    clf_pf.predict([1, 1])


def save_model(model, model_path='model.pkl'):
    """ Serialize the trained model to a file. """
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)


