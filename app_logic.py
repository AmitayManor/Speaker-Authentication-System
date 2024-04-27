from db_process import *

""" Global Variables """
FINAL_THRESHOLD = 0.99

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


def run_threshold_computation():
    compute_data_for_db(THRESHOLD_DB_PATH_AUTHORIZED, THRESHOLD_DB_PATH_IMPOSTER, THRESHOLD_DB_FILE_JSON)
    compute_data_for_db(THRESHOLD_DB_PATH_TEST_AUTHORIZED, THRESHOLD_DB_PATH_TEST_IMPOSTER, THRESHOLD_DB_TEST_FILE_JSON)
    compute_extracted_data(THRESHOLD_DB_FILE_JSON, THRESHOLD_DB_TEST_FILE_JSON, SAVE_THRESHOLDS_MATRIX)


def run_final_db_computation():
    """ Ted-Lium DB"""
    print("calculate test files")
    compute_data_for_db(FINAL_DB_TEST_AUTHORIZED, FINAL_DB_TEST_IMPOSTER, FINAL_DB_FOLDER_TEST_JSON)
    print("calculate train files")
    compute_data_for_db(FINAL_DB_TRAIN_AUTHORIZED, FINAL_DB_TRAIN_IMPOSTER, FINAL_DB_FOLDER_TRAIN_JSON)
    compute_extracted_data(FINAL_DB_FOLDER_TRAIN_JSON, FINAL_DB_FOLDER_TEST_JSON, FINAL_DB_FOLDER_CON_MAT_FILE)


def run_user_computation():
    """ With Max and Amitay recordings """
    """print("calculate test files")
    compute_data_for_db(FINAL_DB_TEST_AUTHORIZED, FINAL_DB_TEST_IMPOSTER, FINAL_DB_FOLDER_TEST_JSON)
    print("calculate train files")
    compute_data_for_db(FINAL_DB_TRAIN_AUTHORIZED, FINAL_DB_TRAIN_IMPOSTER, FINAL_DB_FOLDER_TRAIN_JSON)"""
    compute_extracted_data_from_user(FINAL_DB_FOLDER_TRAIN_JSON, FINAL_DB_FOLDER_TEST_JSON,FINAL_DB_FOLDER_FILE_AMITAY, FINAL_THRESHOLD)

