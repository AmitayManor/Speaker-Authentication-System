from features_extraction import *
from pre_proccess import *
from file_ops import *
import json
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, DetCurveDisplay, det_curve, accuracy_score, f1_score

def confusion_mat_values(tn, fp, fn, tp):
    tn_perc = int(tn / (tn + fp) * 100)
    fp_perc = int(fp / (tn + fp) * 100)
    fn_perc = int(fn / (fn + tp) * 100)
    tp_perc = int(tp / (fn + tp) * 100)

    return tn_perc, fp_perc, fn_perc, tp_perc


def extract_feature_from_db(path):
    data = process_audio_folder(path)
    extracted_features = extract_features_from_processed_data(data)
    return extracted_features


def save_data_to_json(data, path, authorized):
    for x, y in data.items():
        write_features_to_json(y, x, authorized, path)


def write_features_to_json(file_features, file_name, is_authorized, output_file):
    data = {
        'file_name': file_name,
        'is_authorized': is_authorized,
        'features': file_features.tolist() if isinstance(file_features, np.ndarray) else file_features
    }

    with open(output_file, 'a') as f:
        f.write(json.dumps(data) + '\n')

    print(f"Data written to {output_file} successfully.")


def read_features_and_labels(jsonl_file):
    X = []
    y = []
    with open(jsonl_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            X.append(data['features'])
            y.append(1 if data['is_authorized'] else 0)

    return X, y


def train_classifier(X, y):
    clf_pf = GaussianNB(priors=[0.2, 0.8])
    clf_pf.fit(X, y)

    return clf_pf


def train_incremental_classifier(X, y, y_input):
    clf_pf = GaussianNB(priors=[0.2,0.8])
    clf_pf.partial_fit(X, y, np.unique(y_input))
    return clf_pf


def measure_classifier(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_pred, y_test)
    f1 = f1_score(y_pred, y_test, avarage="weighted")

    return acc, f1


def predict_and_round(clf, X_test):
    y_pred = clf.predict_proba(X_test)[:, 1]

    for i in range(len(y_pred)):
        y_pred[i] = y_pred[i] - (y_pred[i] % 0.001)

    y_pred_arr = np.array(y_pred)

    return y_pred_arr


def plot_det_curve(clf, X_test, y_test):
    fig, ax_det = plt.subplots(1, 1, figsize=(11, 5))
    DetCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax_det)
    ax_det.set_title("Detection Error Tradeoff (DET) curves")
    ax_det.grid(linestyle="--")
    plt.legend()
    plt.show()


def compute_con_mat(y_test, y_pred):
    fpr, fnr, thresholds = det_curve(y_test, y_pred)
    return thresholds


def measure_model(y_test, y_pred, threshold, file_name):
    y_test_arr = np.array(y_test)
    bool_arr = np.copy(y_pred)

    for j in range(len(y_pred)):
        if y_pred[j] >= threshold:
            bool_arr[j] = 1
        else:
            bool_arr[j] = 0

    tn, fp, fn, tp = confusion_matrix(y_test_arr, bool_arr).ravel()
    tn_perc, fp_perc, fn_perc, tp_perc = confusion_mat_values(tn, fp, fn, tp)
    acc = (tp+tn)/(tp+tn+fp+fn)
    str1 = f"\nAccuracy: {acc} \n[tn={tn}, fp={fp}\nfn={fn}, tp={tp}]\n"
    str2 = f"[tn={tn_perc}%, fp={fp_perc}%\nfn={fn_perc}%, tp={tp_perc}%]\n"

    append_to_file(file_name, str1)
    append_to_file(file_name, str2)


def test_thresholds(y_test, y_pred, thresholds, file_name):
    y_test_arr = np.array(y_test)
    bool_arr = np.copy(y_pred)

    for i in range(len(thresholds)):
        for j in range(len(y_pred)):
            if y_pred[j] >= thresholds[i]:
                bool_arr[j] = 1
            else:
                bool_arr[j] = 0
        tn, fp, fn, tp = confusion_matrix(y_test_arr, bool_arr).ravel()
        tn_perc, fp_perc, fn_perc, tp_perc = confusion_mat_values(tn, fp, fn, tp)
        str1 = f"\nThreshold: {thresholds[i]} \n[tn={tn}, fp={fp}\nfn={fn}, tp={tp}]\n"
        str2 = f"\nThreshold: {thresholds[i]} \n[tn={tn_perc}%, fp={fp_perc}%\nfn={fn_perc}%, tp={tp_perc}%]\n"

        append_to_file(file_name, str1)
        append_to_file(file_name, str2)


def compute_data_for_db(auth_path_data, impo_path_data, path_save):

    # Extract features from db
    data_authorized = extract_feature_from_db(auth_path_data)
    data_imposter = extract_feature_from_db(impo_path_data)

    # Save data into .txt in JSON format
    save_data_to_json(data_authorized, path_save, True)
    save_data_to_json(data_imposter, path_save, False)


def compute_extracted_data(train_file, test_file, save_file):
    X, y = read_features_and_labels(train_file)
    X_test, y_test = read_features_and_labels(test_file)
    clf = train_classifier(X, y)
    y_pred = predict_and_round(clf, X_test)
    thresholds = compute_con_mat(y_test, y_pred)
    test_thresholds(y_test, y_pred, thresholds, save_file)
    plot_det_curve(clf, X_test, y_test)


def compute_extracted_data_from_user(train_file, test_file, save_file, threshold):
    X, y = read_features_and_labels(train_file)
    X_test, y_test = read_features_and_labels(test_file)
    clf = train_classifier(X, y)
    y_pred = predict_and_round(clf, X_test)
    measure_model(y_test, y_pred, threshold, save_file)


def predict_input(db_file, input_file):
    X, y = read_features_and_labels(db_file)
    X_input, y_input = read_features_and_labels(input_file)
    clf = train_incremental_classifier(X, y, y_input)
    y_pred_input = predict_and_round(clf, X_input)
    """"need to continue"""



def compute_user_data(auth_path_data, path_save):
    data_authorized = extract_feature_from_db(auth_path_data)
    save_data_to_json(data_authorized, path_save, True)