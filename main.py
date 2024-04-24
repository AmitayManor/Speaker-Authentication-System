from features_extraction import compute_lpcc, lpc_to_cepstral
import numpy as np
from pre_proccess import process_audio_folder
from features_extraction import extract_features_from_processed_data, plot_lpcc_features
import json
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import det_curve, DetCurveDisplay


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

    clf_pf = GaussianNB(priors=[0.5, 0.5])
    clf_pf.fit(X, y)

    return clf_pf


def compute_DET_confusion(X_test, y_test):
    y_pred = clf.predict(X_test)

    # Step 2: Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    """# Step 3: Plot confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()"""

    # Step 4: Compute DET curve
    fpr, fnr, _ = det_curve(y_test, y_pred)
    display = DetCurveDisplay(fpr=fpr, fnr=fnr, estimator_name="GaussianNB")
    display.plot()
    plt.title('DET Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('False Negative Rate')
    plt.show()


def mohamad_def(clf, X_test, y_test):
    fig, ax_det = plt.subplots(1, 1, figsize=(11, 5))
    DetCurveDisplay.from_estimator(clf, X_test,y_test, ax=ax_det)
    ax_det.set_title("Detection Error Tradeoff (DET) curves")
    ax_det.grid(linestyle="--")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    """path = "TEDLIUM_release1\\db_for_threshold\\thershold_db"
    data = process_audio_folder(path)
    extracted_features = extract_features_from_processed_data(data)

    for x, y in extracted_features.items():
        write_features_to_json(y, x, False, 'TEDLIUM_release1\\db_for_threshold\\thershold_db\\out_threshold.txt')
    

    path = "TEDLIUM_release1\\db_for_threshold\\threshold_db_test"
    data = process_audio_folder(path)
    extracted_features = extract_features_from_processed_data(data)

    for x, y in extracted_features.items():
        write_features_to_json(y, x, False, 'TEDLIUM_release1\\db_for_threshold\\threshold_db_test\\out_threshold_test.txt')
    """

    X_test, y_test = read_features_and_labels('TEDLIUM_release1\\db_for_threshold\\threshold_db_test\\out_threshold_test.txt')
    X, y = read_features_and_labels('TEDLIUM_release1\\db_for_threshold\\thershold_db\\out_threshold.txt')


    print("Train Data")
    clf = train_classifier(X, y)

    y_pred = clf.predict(X_test)


    fig, ax_det = plt.subplots(1, 1, figsize=(11, 5))
    #DetCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax_det)
    DetCurveDisplay.from_predictions(clf, y_test, y_pred)
    ax_det.set_title("Detection Error Tradeoff (DET) curves")
    ax_det.grid(linestyle="--")
    plt.legend()
    plt.show()

    """print((clf.predict(X_test[0:9])))
    print("Other")
    print(clf.predict(X_test[10::]))

    print("Check for Dan Dennett")
    data = clf.predict_proba(X_test[0:9])
    formatted_data = [[f"{num:.3f}" for num in sublist] for sublist in data]

    # Print formatted data
    for sublist in formatted_data:
        print(sublist)

    print("Others")
    data_test = clf.predict_proba(X_test[10::])
    formatted_data_test = [[f"{num:.6f}" for num in sublist_test] for sublist_test in data_test]

    # Print formatted data
    for sublist_test in formatted_data_test:
        print(sublist_test)"""


    """print("Check for other Speaker")
    test_data = clf.predict_proba(X_test[9::])
    rounded_data_test = np.round(test_data, 4)
    print(rounded_data_test)
    print(clf.predict(X_test[9::]))"""
