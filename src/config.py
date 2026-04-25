COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty"
]

RAW_FEATURE_COUNT = len(COLUMNS) - 2

CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]

TRAIN_PATH = "data/KDDTrain+.txt"
TEST_PATH = "data/KDDTest+.txt"

MODEL_DIR = "models"
REPORT_DIR = "reports"

SCALER_PATH = "models/scaler.pkl"
ENCODERS_PATH = "models/encoders.pkl"
FEATURE_SELECTOR_PATH = "models/feature_selector.pkl"
RANDOM_FOREST_MODEL_PATH = "models/random_forest_model.pkl"
NEURAL_NETWORK_MODEL_PATH = "models/neural_network_model.pkl"
XGBOOST_MODEL_PATH = "models/xgboost_model.pkl"
MODEL_METADATA_PATH = "models/model_metadata.json"
TUNING_RESULTS_PATH = "reports/tuning_results.csv"
