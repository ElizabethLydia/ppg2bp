import torch
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
LEARNING_RATE = 1e-5
BATCH_SIZE = 32
NUM_EPOCHS = 100
PATIENCE = 10

# Centralized output base under autodl-tmp
OUTPUT_BASE = "/root/autodl-tmp/ppg2bp"

SHARED_DATA_DIR = "/root/shared/PhysioNet2025/" 
# Store all pipeline artifacts under OUTPUT_BASE to keep workspace clean
RAW_DATA_DIR = os.path.join(OUTPUT_BASE, "data/raw/")
PROCESSED_DATA_DIR = os.path.join(OUTPUT_BASE, "data/processed/")
RESULTS_DIR = os.path.join(OUTPUT_BASE, "results/")
SAVED_MODELS_DIR = os.path.join(OUTPUT_BASE, "saved_models/")
LOG_DIR = os.path.join(OUTPUT_BASE, "logs/")
MOVED_DATA_DIR = os.path.join(OUTPUT_BASE, "moved/")
SQI_REPORT_DIR = os.path.join(OUTPUT_BASE, "sqi_reports/")
SUBJECT_OVERVIEW_DIR = os.path.join(OUTPUT_BASE, "subject_overviews/")
RUNS_DIR = os.path.join(OUTPUT_BASE, "runs/")  # each training run will be placed under runs/<run_name_timestamp>/

# Ensure directories exist at import time (safe-guard)
for _d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, SAVED_MODELS_DIR, LOG_DIR, MOVED_DATA_DIR, SQI_REPORT_DIR, SUBJECT_OVERVIEW_DIR, RUNS_DIR]:
	try:
		os.makedirs(_d, exist_ok=True)
	except Exception:
		pass

ALL_SUBJECTS = ['00023', '00017', '00041','00042','00043','00044', '00016','00045','00046','00047','00048','00049','00050','00051','00052', '00003','00053','00054','00055','00056','00057','00058','00062','00063', '00064','00065','00066','00071','00072', '00073','00074','00075','00076','00077','00078','00079','00080','00081', '00082','00083','00084','00085','00086','00087','00088','00089','00090', '00091','00092','00093','00094','00095','00096','00097','00098','00099', '00100','00101','00102','00103','00104','00105','00106','00107','00108', '00109','00110','00111','00112']
# 44个训练
TRAIN_SUBJECTS = ['00023', '00017', '00041','00042','00043','00044','00016', '00045','00046','00047','00048','00049','00050','00051','00052','00003','00053', '00054','00055','00056','00057','00058','00062','00063','00064','00065','00066','00072','00073','00074','00076', '00077','00078','00079','00080','00081','00082','00083','00084','00085','00086', '00087','00088','00089'] 
# 13个验证
VALID_SUBJECTS = ['00090','00091','00092','00093','00094','00095','00096','00097', '00098','00100','00101','00102','00104']
# 7个测试
TEST_SUBJECTS = ['00105','00106','00107','00108','00109','00110','00112']

SENSORS_TO_USE = ['sensor2', 'sensor3', 'sensor4', 'sensor5']

BP_SAMPLING_RATE = 1000
SENSOR_SAMPLING_RATE = 110
TARGET_SAMPLING_RATE = 100

WINDOW_SECONDS = 10
WINDOW_SIZE = TARGET_SAMPLING_RATE * WINDOW_SECONDS
STRIDE_SECONDS = 1
STRIDE_SIZE = TARGET_SAMPLING_RATE * STRIDE_SECONDS

IN_CHANNELS = len(SENSORS_TO_USE)
OUTPUT_POINTS = WINDOW_SIZE

BP_FILTER_LOW = 0.5 
BP_FILTER_HIGH = 3 

PPG_FILTER_LOW = 0.5  
PPG_FILTER_HIGH = 3.0 

OMRON_CSV_PATH = "./data/omron.csv" 
BP_CORRECTION_THRESHOLD = 10 
PULSE_PRESSURE_DIFF_THRESHOLD = 10.0
PEAK_MIN_DISTANCE = 0.5 * TARGET_SAMPLING_RATE 

# -------- Robust loss options (defaults keep original behavior) --------
SBP_DBP_MODE = "mse"   # one of: "mse", "huber", "mix"
HUBER_DELTA = 1.0       # huber threshold (mmHg)
MIX_ALPHA = 0.5         # weight for MSE in MSE+MAE mix

# -------- SQI options (training-time consistency control) --------
USE_SQI = True                # enable TD–FD consistency weighting in training
SQI_STRATEGY = "pf"          # 'pf': |HR_time - HR_freq| (recommended); 'pt': reserved for true-HR
SQI_REPORT_ONLY = True        # preprocess-only reporting flag (kept for compatibility)
SQI_MODE = "weight"          # 'weight' or 'filter'
SQI_THRESHOLD = 0.4           # sample kept if weight>=threshold when mode='filter'
CHANNEL_SQI_THRESHOLD = 0.5   # preprocess-only per-channel threshold (kept for compatibility)
CHANNEL_PASS_K = 2            # top-k channels used to form a sample weight
SAMPLE_WEIGHT_MIN = 0.2       # minimal sample weight in 'weight' mode

# -------- SQI detailed export options --------
# Export per-window, per-channel SQI components into CSV for analysis (no behavior change)
SQI_EXPORT_DETAIL = True            # write detailed per-window-per-channel SQI CSVs
SQI_EXPORT_PER_CHANNEL = True       # include per-channel component columns
SQI_INCLUDE_SNR = True              # compute SNR in 0.5–3.0 Hz band

# SNR configuration (Hz)
SNR_BAND_LOW = 0.5
SNR_BAND_HIGH = 3.0
SNR_HR_RANGE = (0.7, 2.5)           # expected heart-rate peak search range
SNR_PEAK_NEIGHBOR = 0.12            # +/- window around f0 considered as signal band

# -------- HR consistency options --------
HR_MIN_BPM = 50
HR_MAX_BPM = 200
HR_TOLERANCE_BPM = 5                # |HR_peak - HR_fft| <= tolerance -> valid
PEAK_MIN_COUNT = 3                  # minimal peaks within a window to trust HR
SQI_SUBJECT_OVERVIEW = True         # render per-subject best/worst window figure