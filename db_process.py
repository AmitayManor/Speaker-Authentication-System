from features_extraction import *
from pre_proccess import *
from file_ops import *
import json
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, DetCurveDisplay, det_curve
import matplotlib.pyplot as plt


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


def train_classifier(X, y, prior):
    clf_pf = GaussianNB(priors=prior)
    clf_pf.fit(X, y)

    return clf_pf


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
    acc = compute_accuracy(tn, fp, fn, tp)
    str1 = f"\nAccuracy: {acc} \n[tn={tn}, fp={fp}\nfn={fn}, tp={tp}]\n"
    str2 = f"[tn={tn_perc}%, fp={fp_perc}%\nfn={fn_perc}%, tp={tp_perc}%]\n"

    append_to_file(file_name, str1)
    append_to_file(file_name, str2)


def compute_accuracy(tn, fp, fn, tp):
    acc = (tp+tn)/(tp+tn+fp+fn)
    return acc


def plot_fp_vs_threshold(thresholds, fp):
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, fp, marker='o', linestyle='-', color='b')
    plt.title('False-Positive Level vs. Threshold Value')
    plt.xlabel('Threshold Value')
    plt.ylabel('False-Positive Level')
    plt.grid(True)
    plt.show()


def plot_accuracy_vs_threshold(thresholds, accuracies, max_accuracy):
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, accuracies, marker='o', linestyle='-', color='b')

    max_index = accuracies.index(max_accuracy)
    max_threshold = thresholds[max_index]

    plt.scatter([max_threshold], [max_accuracy], color='red')
    plt.annotate(f'Threshold: {max_threshold}', (max_threshold, max_accuracy), textcoords="offset points",
                 xytext=(0, 10), ha='center')

    plt.title('Accuracy Level vs. Threshold Value')
    plt.xlabel('Threshold Value')
    plt.ylabel('Accuracy Level')
    plt.grid(True)
    plt.show()


def test_thresholds(y_test, y_pred, thresholds, file_name):
    y_test_arr = np.array(y_test)
    bool_arr = np.copy(y_pred)
    accuracies = []
    fpr = []
    for i in range(len(thresholds)):
        for j in range(len(y_pred)):
            if y_pred[j] >= thresholds[i]:
                bool_arr[j] = 1
            else:
                bool_arr[j] = 0
        tn, fp, fn, tp = confusion_matrix(y_test_arr, bool_arr).ravel()
        tn_perc, fp_perc, fn_perc, tp_perc = confusion_mat_values(tn, fp, fn, tp)
        acc = compute_accuracy(tn, fp, fn, tp)
        accuracies.append(acc)
        fpr.append(fp)
        str1 = f"\nThreshold: {thresholds[i]} \n[tn={tn}, fp={fp}\nfn={fn}, tp={tp}]\n"
        str2 = f"\nThreshold: {thresholds[i]} \n[tn={tn_perc}%, fp={fp_perc}%\nfn={fn_perc}%, tp={tp_perc}%]\n"

        append_to_file(file_name, str1)
        append_to_file(file_name, str2)

    max_acc = max_accuracy(accuracies)
    plot_accuracy_vs_threshold(thresholds, accuracies, max_acc)
    plot_fp_vs_threshold(thresholds, fpr)


def max_accuracy(accuracies):
    max_acc_index = np.argmax(accuracies)
    max_acc = accuracies[max_acc_index]
    return max_acc


def compute_data_for_db(auth_path_data, impo_path_data, path_save):

    data_authorized = extract_feature_from_db(auth_path_data)
    data_imposter = extract_feature_from_db(impo_path_data)

    save_data_to_json(data_authorized, path_save, True)
    save_data_to_json(data_imposter, path_save, False)


def compute_extracted_data(train_file, test_file, save_file, prior):
    X, y = read_features_and_labels(train_file)
    X_test, y_test = read_features_and_labels(test_file)
    clf = train_classifier(X, y, prior)
    y_pred = predict_and_round(clf, X_test)
    thresholds = compute_con_mat(y_test, y_pred)
    test_thresholds(y_test, y_pred, thresholds, save_file)
    plot_det_curve(clf, X_test, y_test)


def compute_extracted_data_from_user(train_file, test_file, save_file, threshold, prior):
    X, y = read_features_and_labels(train_file)
    X_test, y_test = read_features_and_labels(test_file)
    clf = train_classifier(X, y, prior)
    y_pred = predict_and_round(clf, X_test)
    measure_model(y_test, y_pred, threshold, save_file)


def compute_extracted_data_priors(train_file, test_file, save_file):
    X, y = read_features_and_labels(train_file)
    X_test, y_test = read_features_and_labels(test_file)
    check_priors(X, y, X_test, y_test, save_file)


def compute_priors(prior:list, X, y):
    clf = GaussianNB(priors=prior)
    clf.fit(X, y)
    return clf


def check_priors(X, y, X_test, y_test, path):
    prior_list = [[0.01, 0.99], [0.05, 0.95], [0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5]]
    for i in range(len(prior_list)):
        clf = compute_priors(prior_list[i], X, y)
        y_pred = clf.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        tn_perc, fp_perc, fn_perc, tp_perc = confusion_mat_values(tn, fp, fn, tp)
        str1 = f"\nPrior: {prior_list[i]} \n[tn={tn}, fp={fp}\nfn={fn}, tp={tp}]\n"
        str2 = f"\nThreshold: {prior_list[i]} \n[tn={tn_perc}%, fp={fp_perc}%\nfn={fn_perc}%, tp={tp_perc}%]\n"
        append_to_file(path, str1)
        append_to_file(path, str2)