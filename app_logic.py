from db_process import *

""" Global Variables """
FINAL_THRESHOLD = 0.99
FINAL_PRIOR = [0.2, 0.8]


"""" Threshold related paths """
THRESHOLD_DB_PATH_AUTHORIZED = "TEDLIUM_release1\\db_for_threshold\\thershold_db\\authorized"
THRESHOLD_DB_PATH_IMPOSTER = "TEDLIUM_release1\\db_for_threshold\\thershold_db\\imposter"
THRESHOLD_DB_PATH_TEST_AUTHORIZED = "TEDLIUM_release1\\db_for_threshold\\threshold_db_test\\authorized"
THRESHOLD_DB_PATH_TEST_IMPOSTER = "TEDLIUM_release1\\db_for_threshold\\threshold_db_test\\imposter"
THRESHOLD_DB_PATH_JSON = "TEDLIUM_release1\\db_for_threshold\\threshold_db"
THRESHOLD_DB_PATH_TEST_JSON = "TEDLIUM_release1\\db_for_threshold\\threshold_db_test"
THRESHOLD_DB_FILE_JSON = "TEDLIUM_release1\\db_for_threshold\\thershold_db\\extracted_feature_train.txt"
THRESHOLD_DB_TEST_FILE_JSON = "TEDLIUM_release1\\db_for_threshold\\threshold_db_test\\extracted_feature_test.txt"
SAVE_THRESHOLDS_MATRIX = "TEDLIUM_release1\\db_for_threshold\\thresholds_confusion_matrix.txt"


"""" Final DB related paths """
FINAL_DB_FOLDER = "TEDLIUM_release1\\final_db"
FINAL_DB_TEST_AUTHORIZED = "TEDLIUM_release1\\final_db\\final_db_test\\authorized"
FINAL_DB_TEST_IMPOSTER = "TEDLIUM_release1\\final_db\\final_db_test\\imposter"
FINAL_DB_FOLDER_TEST_JSON = "TEDLIUM_release1\\final_db\\final_db_test\\extracted_features_test.txt"
FINAL_DB_TRAIN_AUTHORIZED = "TEDLIUM_release1\\final_db\\final_db_train\\authorized"
FINAL_DB_TRAIN_IMPOSTER = "TEDLIUM_release1\\final_db\\final_db_train\\imposter"
FINAL_DB_FOLDER_TRAIN_JSON = "TEDLIUM_release1\\final_db\\final_db_train\\extracted_features_train.txt"
FINAL_DB_FOLDER_CON_MAT_FILE = "TEDLIUM_release1\\final_db\\final_db_confusion_matrix"
FINAL_DB_FOLDER_FILE_MAX = "TEDLIUM_release1\\final_db\\final_db_confusion_matrix_max"
FINAL_DB_FOLDER_FILE_AMITAY = "TEDLIUM_release1\\final_db\\final_db_confusion_matrix_amitay"
FINAL_DB_NEW_USER_DATA_FILE = "TEDLIUM_release1\\final_db\\new_authorized_user_data.txt"


"""" Priors related paths """
PRIOR_DB_TRAIN_JSON_FILE = "TEDLIUM_release1\\db_for_priors\\db_for_priors_train\\prior_db_train_data.txt"
PRIOR_DB_AUTHORIZED_TRAIN = "TEDLIUM_release1\\db_for_priors\\db_for_priors_train\\authorized"
PRIOR_DB_IMPOSTER_TRAIN = "TEDLIUM_release1\\db_for_priors\\db_for_priors_train\\imposter"
PRIOR_DB_TEST_JSON_FILE = "TEDLIUM_release1\\db_for_priors\\db_for_priors_test\\prior_db_test_data.txt"
PRIOR_DB_AUTHORIZED_TEST = "TEDLIUM_release1\\db_for_priors\\db_for_priors_test\\authorized"
PRIOR_DB_IMPOSTER_TEST = "TEDLIUM_release1\\db_for_priors\\db_for_priors_test\\imposter"
PRIOR_CON_MAT_PATH = "TEDLIUM_release1\\db_for_priors\\prior_db_con_mat.txt"


def test_priors():
    compute_data_for_db(PRIOR_DB_AUTHORIZED_TEST, PRIOR_DB_IMPOSTER_TEST, PRIOR_DB_TEST_JSON_FILE)
    compute_data_for_db(PRIOR_DB_AUTHORIZED_TRAIN, PRIOR_DB_IMPOSTER_TRAIN, PRIOR_DB_TRAIN_JSON_FILE)
    compute_extracted_data_priors(PRIOR_DB_TRAIN_JSON_FILE, PRIOR_DB_TEST_JSON_FILE, PRIOR_CON_MAT_PATH)


def run_threshold_computation():
    compute_data_for_db(THRESHOLD_DB_PATH_AUTHORIZED, THRESHOLD_DB_PATH_IMPOSTER, THRESHOLD_DB_FILE_JSON)
    compute_data_for_db(THRESHOLD_DB_PATH_TEST_AUTHORIZED, THRESHOLD_DB_PATH_TEST_IMPOSTER, THRESHOLD_DB_TEST_FILE_JSON)
    compute_extracted_data(THRESHOLD_DB_FILE_JSON, THRESHOLD_DB_TEST_FILE_JSON, SAVE_THRESHOLDS_MATRIX, FINAL_PRIOR)


def run_final_db_computation():
    """ Ted-Lium DB """
    compute_data_for_db(FINAL_DB_TEST_AUTHORIZED, FINAL_DB_TEST_IMPOSTER, FINAL_DB_FOLDER_TEST_JSON)
    compute_data_for_db(FINAL_DB_TRAIN_AUTHORIZED, FINAL_DB_TRAIN_IMPOSTER, FINAL_DB_FOLDER_TRAIN_JSON)
    compute_extracted_data(FINAL_DB_FOLDER_TRAIN_JSON, FINAL_DB_FOLDER_TEST_JSON, FINAL_DB_FOLDER_CON_MAT_FILE, FINAL_PRIOR)


def run_user_computation():
    """ With Max and Amitay recordings """
    compute_data_for_db(FINAL_DB_TEST_AUTHORIZED, FINAL_DB_TEST_IMPOSTER, FINAL_DB_FOLDER_TEST_JSON)
    compute_data_for_db(FINAL_DB_TRAIN_AUTHORIZED, FINAL_DB_TRAIN_IMPOSTER, FINAL_DB_FOLDER_TRAIN_JSON)
    compute_extracted_data_from_user(FINAL_DB_FOLDER_TRAIN_JSON, FINAL_DB_FOLDER_TEST_JSON,FINAL_DB_FOLDER_FILE_AMITAY, FINAL_THRESHOLD, FINAL_PRIOR)

