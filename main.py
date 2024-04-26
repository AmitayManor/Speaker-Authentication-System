from features_extraction import compute_lpcc, lpc_to_cepstral
import numpy as np
from pre_proccess import process_audio_folder
from features_extraction import extract_features_from_processed_data, plot_lpcc_features
import json
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, det_curve
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from sklearn.metrics import DetCurveDisplay, RocCurveDisplay
from db_process import *
from file_ops import *

CON_MAT_FILE = 'con_mat_txt.txt'


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


def write_features_to_json(file_features: object, file_name: object, is_authorized: object,
                           output_file: object) -> object:
    """
    Write the extracted features and metadata to a JSON-formatted text file.

    Args:
        file_features (list or np.array): The extracted features for the audio file.
        file_name (str): The name of the audio file.
        is_authorized (bool): True if the speaker is authorized, False if an imposter.
        output_file (str): The path to the output text file where data will be written in JSON format.
        :rtype: object
    """
    # Create a dictionary with the data
    data = {
        'file_name': file_name,
        'is_authorized': is_authorized,
        'features': file_features.tolist() if isinstance(file_features, np.ndarray) else file_features
    }

    # Append to the .txt file as a JSON Line
    with open(output_file, 'a') as f:  # Open in append mode
        f.write(json.dumps(data) + '\n')  # Write JSON object followed by a newline

    print(f"Data written to {output_file} successfully.")

def write_string_to_file(str, file):
    f = open(file, "a")
    f.write(str)
    f.close()


def read_features_and_labels(jsonl_file):
    """
    Read features and authorization labels from a JSON Lines formatted text file.

    Args:
        jsonl_file (str): The path to the JSON Lines text file containing the data.

    Returns:
        X (list of lists): The list of feature lists.
        y (list of int): The list of labels where authorized is 1 and imposter is 0.
    """
    X = []  # Initialize the list for features
    y = []  # Initialize the list for labels

    with open(jsonl_file, 'r') as file:
        for line in file:
            data = json.loads(line)  # Parse the JSON object
            X.append(data['features'])  # Add the feature list to X
            y.append(1 if data['is_authorized'] else 0)  # Convert True/False to 1/0 and add to y

    return X, y


def train_classifier(X, y):
    """ Train a Gaussian Naive Bayes classifier.
    x = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    y = np.array([1, 1, 1, 0, 0, 0])"""

    clf_pf = GaussianNB(priors=[0.2, 0.8])
    clf_pf.fit(X, y)

    return clf_pf


def compute_DET_confusion(y_pred, y_test):
    # Step 2: Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Step 3: Plot confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

THRESHOLD_DB_PATH_AUTHORIZED = "TEDLIUM_release1\\db_for_threshold\\thershold_db\\authorized"
THRESHOLD_DB_PATH_IMPOSTER = "TEDLIUM_release1\\db_for_threshold\\thershold_db\\imposter"
THRESHOLD_DB_PATH_TEST_AUTHORIZED = "TEDLIUM_release1\\db_for_threshold\\threshold_db_test\\authorized"
THRESHOLD_DB_PATH_TEST_IMPOSTER = "TEDLIUM_release1\\db_for_threshold\\threshold_db_test\\imposter"

THRESHOLD_DB_PATH_JSON = "TEDLIUM_release1\\db_for_threshold\\threshold_db"
THRESHOLD_DB_PATH_TEST_JSON = "TEDLIUM_release1\\db_for_threshold\\threshold_db_test"

THRESHOLD_DB_FILE_JSON = "TEDLIUM_release1\\db_for_threshold\\thershold_db\\extracted_feature_train.txt"
THRESHOLD_DB_TEST_FILE_JSON = "TEDLIUM_release1\\db_for_threshold\\threshold_db_test\\extracted_feature_test.txt"
SAVE_THRESHOLDS_MATRIX = "TEDLIUM_release1\\db_for_threshold\\thershold_db\\thresholds_confusion_matrix.txt"
if __name__ == "__main__":

    """# Train
    compute_data_for_db(THRESHOLD_DB_PATH_AUTHORIZED,THRESHOLD_DB_PATH_IMPOSTER,THRESHOLD_DB_FILE_JSON,THRESHOLD_DB_FILE_JSON)
    
    # Test
    compute_data_for_db(THRESHOLD_DB_PATH_TEST_AUTHORIZED, THRESHOLD_DB_PATH_TEST_IMPOSTER, THRESHOLD_DB_TEST_FILE_JSON,THRESHOLD_DB_TEST_FILE_JSON)
    """

    compute_extracted_data(THRESHOLD_DB_FILE_JSON,THRESHOLD_DB_TEST_FILE_JSON,SAVE_THRESHOLDS_MATRIX)

    X_test, y_test = read_features_and_labels(
        'TEDLIUM_release1\\db_for_threshold\\threshold_db_test\\extracted_feature_test.txt')
    X, y = read_features_and_labels('TEDLIUM_release1\\db_for_threshold\\thershold_db\\extracted_feature_train.txt')

    print("Train Data")
    clf = train_classifier(X, y)

    y_pred = clf.predict_proba(X_test)[:, 1]

    for i in range(len(y_pred)):
        y_pred[i] = y_pred[i] - (y_pred[i] % 0.001)

    y_pred_arr = np.array(y_pred)
    print(y_pred_arr)
    y_test_arr = np.array(y_test)
    print(y_test_arr)
    X_test_arr = np.array(X_test)
    fpr, fnr, thresholds = det_curve(y_test_arr, y_pred_arr)

    fig, ax_det = plt.subplots(1, 1, figsize=(11, 5))
    DetCurveDisplay.from_estimator(clf, X_test_arr, y_test_arr, ax=ax_det)
    ax_det.set_title("Detection Error Tradeoff (DET) curves")
    ax_det.grid(linestyle="--")
    plt.legend()
    plt.show()

    arr = np.copy(y_pred_arr)

    for i in range(len(thresholds)):
        for j in range(len(y_pred_arr)):
            if y_pred_arr[j] >= thresholds[i]:
                arr[j] = 1
            else:
                arr[j] = 0
        tn, fp, fn, tp = confusion_matrix(y_test_arr, arr).ravel()
        tn_perc, fp_perc, fn_perc, tp_perc = confusin_mat_values(tn, fp, fn, tp)

        str1 = (f"\nThreshold: {thresholds[i]} \n[tn={tn}, fp={fp}\nfn={fn}, tp={tp}]\n")
        str2 = (f"\nThreshold: {thresholds[i]} \n[tn={tn_perc}%, fp={fp_perc}%\nfn={fn_perc}%, tp={tp_perc}%]\n")
        write_string_to_file(str1, CON_MAT_FILE)
        write_string_to_file(str2, CON_MAT_FILE)