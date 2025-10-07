import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil
import tarfile
from scipy import signal
from scipy.interpolate import interp1d
import sys
import os
import logging
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configure logging
from datetime import datetime
import matplotlib.pyplot as plt
import csv

# Centralized logging under config.LOG_DIR
try:
    os.makedirs(getattr(__import__('config'), 'LOG_DIR', './'), exist_ok=True)
    log_path = os.path.join(getattr(__import__('config'), 'LOG_DIR', './'), f'processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
except Exception:
    log_path = 'processing.log'

logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   filename=log_path,
                   filemode='w')

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def create_subject_best_worst_overview(self, subject: str, subject_results: dict, subject_signals: dict):
    """为单个受试者绘制总图：每个传感器挑选一个“最好窗口”和“最差窗口”。
    质量依据：
      - 优先使用 PT 误差（|峰值HR-真实HR|，越小越好）
      - 若该窗口缺少真实HR则回退用 SNR（越大越好）
      - 若窗口 is_valid=True，给微弱加分（不改变主排序只做tie-break）
    在图上标注：SNR(dB) 与 PTerr(bpm)；若无真实HR则显示 N/A。
    """
    import math
    rows = len(self.sensors)
    cols = 2  # best, worst
    fig, axes = plt.subplots(rows, cols, figsize=(cols*8, rows*3), squeeze=False, sharex=False)
    fig.suptitle(f'{subject} - Best/Worst Quality Windows per Sensor', fontsize=16, fontweight='bold')

    for r, sensor in enumerate(self.sensors):
        # 收集 (exp_id, window_dict, fs, signal)
        candidates = []
        for exp_id, pack in subject_results.items():
            win_list = pack.get('sensor_window_results', {}).get(sensor, [])
            sig_pack = subject_signals.get(exp_id, {}).get(sensor, None)
            if sig_pack is None:
                continue
            fs = int(sig_pack.get('fs', self.default_fs))
            sig = sig_pack.get('signal', None)
            if sig is None:
                continue
            for w in win_list:
                candidates.append((exp_id, w, fs, sig))

        if not candidates:
            for c in range(cols):
                axes[r, c].text(0.5, 0.5, f'{self.sensor_mapping.get(sensor, sensor)}: No Windows',
                                ha='center', va='center', transform=axes[r, c].transAxes)
                axes[r, c].set_axis_off()
            continue

        # 质量评分：优先PT误差（越小越好，用负号变成“越大越好”），缺失则用SNR
        def quality_score(item):
            exp_id, w, fs, sig = item
            true_hr = w.get('true_hr_bpm', np.nan)
            peak_hr = float(w.get('peak_hr_bpm', np.nan))
            snr = float(w.get('snr_db', -1e9))
            if isinstance(true_hr, float) and np.isfinite(true_hr):
                score = -abs(peak_hr - true_hr)  # 误差越小越好
            else:
                score = snr  # 缺真实HR时，用SNR
            # 有效窗口轻微加分（打破并列用）
            if w.get('is_valid', False):
                score += 0.1
            return score

        best_item = max(candidates, key=quality_score)
        worst_item = min(candidates, key=quality_score)

        for c, (title, item) in enumerate([(f'{self.sensor_mapping.get(sensor, sensor)} - BEST', best_item),
                                           (f'{self.sensor_mapping.get(sensor, sensor)} - WORST', worst_item)]):
            exp_id, w, fs, sig = item
            start, end = int(w['start_sample']), int(w['end_sample'])
            seg = np.asarray(sig[start:end], dtype=float)
            t = np.arange(len(seg)) / fs
            filt = self.bandpass_filter(seg, fs=fs)

            ax = axes[r, c]
            ax.plot(t, filt, color='tab:blue' if c == 0 else 'tab:red', linewidth=1.0)

            # 标记峰值（映射到窗口坐标）
            try:
                gp = w.get('global_peak_indices', np.array([], dtype=int))
                local_peaks = gp - start
                local_peaks = local_peaks[(local_peaks >= 0) & (local_peaks < len(filt))]
                if len(local_peaks) > 0:
                    ax.scatter(local_peaks / fs, filt[local_peaks], s=14, c='orange', zorder=5, label='Peaks')
            except Exception:
                pass

            # 准备注释：SNR 和 PT 误差
            true_hr = w.get('true_hr_bpm', np.nan)
            peak_hr = float(w.get('peak_hr_bpm', np.nan))
            fft_hr = float(w.get('fft_hr_bpm', np.nan))
            snr_db = float(w.get('snr_db', np.nan))
            pt_err = abs(peak_hr - true_hr) if (isinstance(true_hr, float) and np.isfinite(true_hr)) else np.nan

            meta = {
                'exp': exp_id,
                'valid': w.get('is_valid', False),
                'HRp': peak_hr,
                'HRf': fft_hr,
                'PTerr': pt_err,
                'SNR(dB)': snr_db,
            }
            subtitle = ", ".join([
                f"{k}={v:.2f}" if isinstance(v, float) and np.isfinite(v) else f"{k}={v}"
                for k, v in meta.items()
            ])
            ax.set_title(f"{title} | {subtitle}", fontsize=10)

            # 右下角文字框：只强调 SNR 和 PTerr
            box_lines = []
            box_lines.append(f"SNR: {snr_db:.2f} dB" if np.isfinite(snr_db) else "SNR: N/A")
            box_lines.append(f"PT err: {pt_err:.2f} bpm" if np.isfinite(pt_err) else "PT err: N/A")
            ax.text(0.98, 0.02, "\n".join(box_lines), transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=9, color='black',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='gray'))

            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Filtered IR')
            ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(self.output_dir, f"{subject}_best_worst_windows.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 保存受试者总览图: {out_path}")
    
def decompress_archives(source_dir):
    logging.debug(f"Scanning for archives in: {source_dir}")
    archives = glob.glob(os.path.join(source_dir, "*.tar")) + glob.glob(os.path.join(source_dir, "*.tar.gz"))
    
    if not archives:
        logging.debug("No archives found to decompress.")
        return

    for archive_path in archives:
        archive_name = os.path.basename(archive_path)
        extracted_folder_name = archive_name.replace(".tar.gz", "").replace(".tar", "")
        extracted_folder_path = os.path.join(source_dir, extracted_folder_name)
        
        if os.path.isdir(extracted_folder_path):
            logging.debug(f"Directory '{extracted_folder_name}' already exists. Skipping.")
            continue
            
        logging.debug(f"Decompressing '{archive_name}'...")
        try:
            with tarfile.open(archive_path, "r:*") as tar:
                tar.extractall(path=source_dir)
            logging.debug("Successfully decompressed.")
        except Exception as e:
            logging.error(f"Error decompressing {archive_name}: {e}")

def organize_raw_data(source_base_dir, dest_dir, subjects_to_process):
    logging.debug(f"Organizing raw data from: {source_base_dir}")
    if not os.path.isdir(source_base_dir):
        logging.error(f"Error: Source directory not found at {source_base_dir}")
        return

    task_mapping = {'1': 'seg1', '7': 'seg7'}

    for subject_id in tqdm(subjects_to_process, desc="Organizing subjects"):
        subject_dest_folder = os.path.join(dest_dir, subject_id)
        os.makedirs(subject_dest_folder, exist_ok=True)
        
        subject_paths = glob.glob(os.path.join(source_base_dir, "*", subject_id))
        
        if not subject_paths:
            logging.warning(f"Warning: No folder found for subject {subject_id}")
            continue
        
        subject_path = subject_paths[0]
        found_files_for_subject = False

        for task_id, seg_name in task_mapping.items():
            task_folder = os.path.join(subject_path, task_id)
            if not os.path.isdir(task_folder):
                continue

            biopac_folder = os.path.join(task_folder, "Biopac")
            
            # Copy BP files
            bp_src_path = os.path.join(biopac_folder, "bp.csv")
            if os.path.exists(bp_src_path):
                bp_dest_path = os.path.join(subject_dest_folder, f"{seg_name}_bp.csv")
                shutil.copy2(bp_src_path, bp_dest_path)
                found_files_for_subject = True
            
            # Copy HR file
            hr_src_path = os.path.join(biopac_folder, "hr.csv")
            if os.path.exists(hr_src_path):
                hr_dest_path = os.path.join(subject_dest_folder, f"{seg_name}_hr.csv")
                shutil.copy2(hr_src_path, hr_dest_path)
            
            # Copy systolic and diastolic BP files
            systolic_files = glob.glob(os.path.join(biopac_folder, "*systolicbp.csv"))
            if systolic_files:
                shutil.copy2(systolic_files[0], os.path.join(subject_dest_folder, f"{seg_name}_systolicbp.csv"))
            
            diastolic_files = glob.glob(os.path.join(biopac_folder, "*diastolic*bp.csv"))
            if diastolic_files:
                shutil.copy2(diastolic_files[0], os.path.join(subject_dest_folder, f"{seg_name}_diastolicbp.csv"))

            hub_folder = os.path.join(task_folder, "HUB")
            if os.path.isdir(hub_folder):
                sensor_files = glob.glob(os.path.join(hub_folder, "*.csv"))
                for sensor_src_path in sensor_files:
                    original_filename = os.path.basename(sensor_src_path)
                    sensor_dest_path = os.path.join(subject_dest_folder, f"{seg_name}_{original_filename}")
                    shutil.copy2(sensor_src_path, sensor_dest_path)
                    found_files_for_subject = True

        if found_files_for_subject:
             logging.debug(f"Organized data for {subject_id}")

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    if len(data) < order * 3: 
        logging.debug(f"Warning: Data too short for filtering ({len(data)} points), skipping filter")
        return data
    
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # 确保频率在有效范围内
    if low <= 0:
        low = 0.001
    if high >= 1:
        high = 0.999
    
    # 检查频率顺序
    if low >= high:
        logging.debug(f"Warning: Invalid frequency range (low={low}, high={high}), skipping filter")
        return data
        
    try:
        b, a = signal.butter(order, [low, high], btype='band')
        filtered_data = signal.filtfilt(b, a, data)
        
        # 检查滤波结果
        if np.all(filtered_data == 0) or np.all(np.isnan(filtered_data)):
            logging.debug("Warning: Filtering resulted in all zeros/NaNs, returning original data")
            return data
            
        logging.debug(f"Filtering: {len(data)} points, range before=({np.min(data):.4f}, {np.max(data):.4f}), after=({np.min(filtered_data):.4f}, {np.max(filtered_data):.4f})")
        return filtered_data
        
    except Exception as e:
        logging.debug(f"Warning: Filtering failed ({e}), returning original data")
        return data

def resample_data(data, timestamps, original_fs, target_fs):
    if len(data) < 2 or len(timestamps) < 2:
        return np.array([]), np.array([])
    
    duration = timestamps[-1] - timestamps[0]
    target_length = int(duration * target_fs)
    
    if target_length < 1:
        return np.array([]), np.array([])
    
    new_timestamps = np.linspace(timestamps[0], timestamps[-1], target_length)
    
    try:
        interp_func = interp1d(timestamps, data, kind='linear', bounds_error=False, fill_value='extrapolate')
        new_data = interp_func(new_timestamps)
        return new_data, new_timestamps
    except Exception as e:
        logging.error(f"Resampling error: {e}")
        return np.array([]), np.array([])

def normalize_signal(data):
    if len(data) == 0:
        return data
    
    # 检查数据是否全为相同值
    if np.all(data == data[0]):
        logging.debug(f"Warning: Signal is constant (value={data[0]}), skipping normalization")
        return data - np.mean(data)  # 只去均值
    
    mean_val = np.mean(data)
    std_val = np.std(data)
    
    if std_val < 1e-8:
        logging.debug(f"Warning: Very small std ({std_val}), using robust normalization")
        # 使用更鲁棒的归一化方法
        median_val = np.median(data)
        mad = np.median(np.abs(data - median_val))
        if mad > 1e-8:
            return (data - median_val) / mad
        else:
            return data - mean_val
    
    normalized = (data - mean_val) / std_val
    logging.debug(f"Normalization: mean={mean_val:.4f}, std={std_val:.4f}, range=({np.min(normalized):.4f}, {np.max(normalized):.4f})")
    return normalized

def load_heart_rate(subject_folder, segment):
    """
    Load heart rate data from hr.csv file
    
    Args:
        subject_folder: Path to subject folder
        segment: Segment name (seg1 or seg7)
    
    Returns:
        Average heart rate in bpm, or None if loading fails
    """
    hr_path = os.path.join(subject_folder, f'{segment}_hr.csv')
    
    if not os.path.exists(hr_path):
        logging.debug(f"HR file not found: {hr_path}")
        return None
    
    try:
        df_hr = pd.read_csv(hr_path, header=None, dtype=str)
        df_hr.columns = ['timestamp', 'hr']
        
        df_hr['hr'] = pd.to_numeric(df_hr['hr'], errors='coerce')
        df_hr.dropna(inplace=True)
        
        if df_hr.empty:
            logging.debug(f"No valid HR data in {hr_path}")
            return None
        
        # Calculate average HR
        avg_hr = df_hr['hr'].mean()
        
        # Check if HR is in reasonable range (30-200 bpm)
        if avg_hr < 30 or avg_hr > 200:
            logging.debug(f"HR out of reasonable range: {avg_hr:.1f} bpm")
            return None
            
        logging.debug(f"Loaded HR data: average {avg_hr:.1f} bpm")
        return avg_hr
        
    except Exception as e:
        logging.debug(f"Error loading HR data: {e}")
        return None

def load_hr_series(subject_folder, segment):
    """
    加载每个分段的 HR 时间序列，返回 DataFrame(columns=[timestamp, hr, rel_time_s])。
    rel_time_s = timestamp - timestamp.iloc[0]
    如果文件缺失或无效，返回 None。
    """
    hr_path = os.path.join(subject_folder, f'{segment}_hr.csv')
    if not os.path.exists(hr_path):
        logging.debug(f"HR series file not found: {hr_path}")
        return None
    try:
        df_hr = pd.read_csv(hr_path, header=None, dtype=str)
        if df_hr.shape[1] < 2:
            logging.debug(f"Invalid HR file format: {hr_path}")
            return None
        df_hr.columns = ['timestamp', 'hr']
        df_hr['timestamp'] = pd.to_numeric(df_hr['timestamp'], errors='coerce')
        df_hr['hr'] = pd.to_numeric(df_hr['hr'], errors='coerce')
        df_hr.dropna(inplace=True)
        if df_hr.empty:
            return None
        df_hr = df_hr.sort_values('timestamp').reset_index(drop=True)
        t0 = df_hr['timestamp'].iloc[0]
        df_hr['rel_time_s'] = df_hr['timestamp'] - t0
        return df_hr
    except Exception as e:
        logging.debug(f"Error loading HR series: {e}")
        return None

def load_omron_data():
    """
    Load Omron BP data from centralized omron.csv file.
    It assumes one row per subject, applying the same BP values to both seg1 and seg7.
    
    Returns:
        A dictionary mapping 'subjectID_segment' to (sbp, dbp) tuples.
        e.g., {'00042_seg1': (107, 77), '00042_seg7': (107, 77)}
    """
    possible_paths = [
        os.path.join(os.path.dirname(config.RAW_DATA_DIR), 'omron.csv'),
        '/root/ppg2bp/data/omron.csv',
    ]
    
    omron_path = None
    for path in possible_paths:
        if os.path.exists(path):
            omron_path = path
            break
    
    if omron_path is None:
        logging.debug("Omron file not found in any expected locations.")
        return {}
    
    logging.debug(f"Loading Omron data from: {omron_path}")
    
    try:
        df_omron = pd.read_csv(omron_path)
        df_omron.columns = [col.lower().strip() for col in df_omron.columns]
        
        required_cols = {'id', 'sbp', 'dbp'}
        if not required_cols.issubset(df_omron.columns):
            logging.error(f"Omron file is missing required columns. Expected: {required_cols}")
            return {}

        omron_data = {}
        for _, row in df_omron.iterrows():
            subject_id_raw = row['id']
            sbp = pd.to_numeric(row['sbp'], errors='coerce')
            dbp = pd.to_numeric(row['dbp'], errors='coerce')
            
            if pd.notna(subject_id_raw) and pd.notna(sbp) and pd.notna(dbp):
                # --- FIX IS HERE ---
                # Convert ID to integer first to remove ".0", then format to 5-digit string.
                subject_id = str(int(subject_id_raw)).strip().zfill(5)
                
                omron_data[f"{subject_id}_seg1"] = (sbp, dbp)
                omron_data[f"{subject_id}_seg7"] = (sbp, dbp)
                logging.debug(f"Loaded and stored Omron data for key '{subject_id}_seg1' and '{subject_id}_seg7'")
        
        return omron_data
        
    except Exception as e:
        logging.error(f"Failed to load or parse Omron data from '{omron_path}': {e}")
        return {}

def load_biopac_bp_values(subject_folder, segment):
    """
    Load Biopac SBP and DBP from separate files
    
    Returns:
        (sbp_mean, dbp_mean) tuple or (None, None) if loading fails
    """
    systolic_path = os.path.join(subject_folder, f'{segment}_systolic_bp.csv')
    diastolic_path = os.path.join(subject_folder, f'{segment}_diastolic_bp.csv')
    
    try:
        # Load systolic BP
        if os.path.exists(systolic_path):
            df_sbp = pd.read_csv(systolic_path, header=None, dtype=str)
            df_sbp.columns = ['timestamp', 'sbp']
            df_sbp['sbp'] = pd.to_numeric(df_sbp['sbp'], errors='coerce')
            df_sbp.dropna(inplace=True)
            # Filter out zero values
            df_sbp = df_sbp[df_sbp['sbp'] != 0]
            sbp_mean = df_sbp['sbp'].mean() if not df_sbp.empty else None
        else:
            logging.debug(f"Systolic BP file not found: {systolic_path}")
            sbp_mean = None
        
        # Load diastolic BP
        if os.path.exists(diastolic_path):
            df_dbp = pd.read_csv(diastolic_path, header=None, dtype=str)
            df_dbp.columns = ['timestamp', 'dbp']
            df_dbp['dbp'] = pd.to_numeric(df_dbp['dbp'], errors='coerce')
            df_dbp.dropna(inplace=True)
            # Filter out zero values
            df_dbp = df_dbp[df_dbp['dbp'] != 0]
            dbp_mean = df_dbp['dbp'].mean() if not df_dbp.empty else None
        else:
            logging.debug(f"Diastolic BP file not found: {diastolic_path}")
            dbp_mean = None
            
        if sbp_mean is not None and dbp_mean is not None:
            logging.debug(f"Biopac BP values for {segment}: SBP={sbp_mean:.1f}, DBP={dbp_mean:.1f} (zeros excluded)")
        
        return sbp_mean, dbp_mean
        
    except Exception as e:
        logging.error(f"Error loading Biopac BP values: {e}")
        return None, None

def calculate_bp_shift_amount(subject_id):
    """
    Calculate BP shift amount based on seg1 Omron vs Biopac comparison
    
    Args:
        subject_id: Subject ID (e.g., '00042')
    
    Returns:
        avg_diff: Average shift amount in mmHg, or None if calculation fails
    """
    # Load all Omron data
    omron_data = load_omron_data()
    
    # Look up seg1 data for this subject
    omron_key = f"{subject_id}_seg1"
    if omron_key not in omron_data:
        logging.debug(f"No Omron data for {subject_id} seg1")
        return None
    
    omron_sbp, omron_dbp = omron_data[omron_key]
    
    # Load Biopac SBP/DBP for seg1
    subject_folder = os.path.join(config.RAW_DATA_DIR, subject_id)
    biopac_sbp, biopac_dbp = load_biopac_bp_values(subject_folder, 'seg1')
    
    if biopac_sbp is None or biopac_dbp is None:
        logging.debug(f"No Biopac SBP/DBP data for {subject_id} seg1")
        return None
    
    # Calculate differences
    sbp_diff = omron_sbp - biopac_sbp
    dbp_diff = omron_dbp - biopac_dbp
    avg_diff = (sbp_diff + dbp_diff) / 2
    
    # Return shift amount if difference is reasonable (within ±10 mmHg)
    if abs(sbp_diff - dbp_diff) <= 10:
        logging.debug(f"Calculated shift amount for {subject_id}: {avg_diff:+.1f} mmHg")
        return avg_diff
    else:
        logging.debug(f"No shift for {subject_id}, delta too large ({sbp_diff - dbp_diff:.1f} mmHg)")
        return None

def apply_bp_calibration(bp_data, subject_id, segment, shift_amount=None):
    """
    Apply BP calibration using pre-calculated shift amount
    
    Args:
        bp_data: BP waveform data
        subject_id: Subject ID (e.g., '00042')
        segment: Segment name (e.g., 'seg1' or 'seg7')
        shift_amount: Pre-calculated shift amount, or None to calculate from seg1
    
    Returns:
        Calibrated BP data and shift amount (for printing)
    """
    # If shift_amount is not provided, calculate it from seg1
    if shift_amount is None:
        shift_amount = calculate_bp_shift_amount(subject_id)
    
    if shift_amount is not None:
        calibrated_bp = bp_data + shift_amount
        print(f"segment {segment}: BP waveform shifted by {shift_amount:+.1f} mmHg")
        return calibrated_bp, shift_amount
    else:
        print(f"segment {segment}: No shift applied")
        return bp_data, None

def save_calibrated_bp(bp_data, original_timestamps, subject_id, segment, moved_data_dir):
    """
    Save calibrated BP data to moved directory in original CSV format
    
    Args:
        bp_data: Calibrated BP waveform data
        original_timestamps: Original timestamps from raw data
        subject_id: Subject ID (e.g., '00042')
        segment: Segment name (e.g., 'seg1')
        moved_data_dir: Base directory for moved data
    """
    # Create subject directory if not exists
    subject_moved_dir = os.path.join(moved_data_dir, subject_id)
    os.makedirs(subject_moved_dir, exist_ok=True)
    
    # Save to CSV with exact same format as original
    moved_bp_path = os.path.join(subject_moved_dir, f'{segment}_bp.csv')
    
    with open(moved_bp_path, 'w') as f:
        for timestamp, bp_value in zip(original_timestamps, bp_data):
            # Format to match original: preserve timestamp precision, round BP to 4 decimal places
            f.write(f"{timestamp:.7f},{bp_value:.4f}\n")
    
    logging.debug(f"Saved calibrated BP data to: {moved_bp_path}")

def find_bp_valleys(bp_data, fs, avg_hr=None):
    """
    找到血压信号的真正波谷（舒张期最低点）
    使用更鲁棒的多步骤方法
    
    Args:
        bp_data: 血压信号数组
        fs: 采样率
        avg_hr: 平均心率 (bpm)，如果提供则用于计算最小间距
    
    Returns:
        valley_indices: 波谷位置的索引数组
    """
    if len(bp_data) < fs:
        return np.array([])
    
    # 步骤1: 预处理 - 轻度平滑去噪但保持波形
    from scipy.ndimage import median_filter
    smoothed_bp = median_filter(bp_data, size=max(3, int(fs * 0.02)))  # 20ms中值滤波
    
    # 步骤2: 估算心率，动态调整参数
    if avg_hr is not None:
        # Use provided heart rate
        estimated_hr = avg_hr
        expected_rr = 60 / estimated_hr
        min_rr_interval = expected_rr * 0.7  # Allow some variation
        max_rr_interval = expected_rr * 1.3
        logging.debug(f"Using provided HR: {estimated_hr:.1f} bpm, RR interval: {expected_rr:.3f}s")
    else:
        # Original FFT-based estimation
        from scipy.fft import fft, fftfreq
        freqs = fftfreq(len(smoothed_bp), 1/fs)
        fft_vals = np.abs(fft(smoothed_bp))
        # 寻找0.5-3Hz范围内的主频率（30-180 bpm）
        valid_freq_mask = (freqs >= 0.5) & (freqs <= 3.0)
        if np.any(valid_freq_mask):
            dominant_freq = freqs[valid_freq_mask][np.argmax(fft_vals[valid_freq_mask])]
            estimated_hr = dominant_freq * 60
            min_rr_interval = max(0.4, 60/180)  # 最快不超过180bpm
            max_rr_interval = min(2.0, 60/40)   # 最慢不低于40bpm
            expected_rr = 60/max(40, min(180, estimated_hr))
        else:
            expected_rr = 1.0  # 默认60bpm
            min_rr_interval = 0.6
            max_rr_interval = 1.5
            estimated_hr = 60
        logging.debug(f"Estimated HR from FFT: {estimated_hr:.1f} bpm")
    
    min_distance = int(min_rr_interval * fs)
    
    # 步骤3: 多层次波谷检测
    # 3.1 粗检测 - 找所有可能的波谷
    rough_valleys, _ = signal.find_peaks(
        -smoothed_bp,
        distance=min_distance // 2,  # 更小的距离
        prominence=np.std(smoothed_bp) * 0.1,  # 更低的阈值
        width=max(2, int(fs * 0.02))  # 最小宽度20ms
    )
    
    if len(rough_valleys) == 0:
        return np.array([])
    
    # 3.2 精细化 - 基于局部最小值和生理约束
    refined_valleys = []
    last_valley_idx = -1
    
    for valley_idx in rough_valleys:
        # 检查与上一个波谷的时间间隔
        if last_valley_idx >= 0:
            time_diff = (valley_idx - last_valley_idx) / fs
            if time_diff < min_rr_interval:  # 太近，跳过
                continue
                
        # 在小窗口内寻找真正的最小值点
        search_window = int(fs * 0.05)  # 50ms窗口
        start_idx = max(0, valley_idx - search_window)
        end_idx = min(len(bp_data), valley_idx + search_window)
        
        local_min_idx = start_idx + np.argmin(bp_data[start_idx:end_idx])
        
        # 验证这确实是一个显著的波谷
        # 检查左右两侧是否都有明显上升
        left_check = max(0, local_min_idx - int(fs * 0.1))   # 左侧100ms
        right_check = min(len(bp_data), local_min_idx + int(fs * 0.1))  # 右侧100ms
        
        left_max = np.max(bp_data[left_check:local_min_idx+1]) if local_min_idx > left_check else bp_data[local_min_idx]
        right_max = np.max(bp_data[local_min_idx:right_check]) if right_check > local_min_idx else bp_data[local_min_idx]
        
        # 波谷深度检查 - 两侧都应该有明显上升
        min_depth = np.std(bp_data) * 0.15  # 相对较小的阈值
        if (left_max - bp_data[local_min_idx]) > min_depth and (right_max - bp_data[local_min_idx]) > min_depth:
            refined_valleys.append(local_min_idx)
            last_valley_idx = local_min_idx
    
    # 步骤4: 最终验证和补充
    if len(refined_valleys) > 1:
        final_valleys = []
        
        for i, valley_idx in enumerate(refined_valleys):
            final_valleys.append(valley_idx)
            
            # 检查是否需要在两个波谷之间补充遗漏的
            if i < len(refined_valleys) - 1:
                next_valley = refined_valleys[i + 1]
                gap_time = (next_valley - valley_idx) / fs
                
                # 如果间隔过大（超过1.8个预期RR间隔），尝试在中间找波谷
                if gap_time > expected_rr * 1.8:
                    mid_start = valley_idx + int(expected_rr * 0.7 * fs)
                    mid_end = next_valley - int(expected_rr * 0.3 * fs)
                    
                    if mid_end > mid_start:
                        mid_valley_idx = mid_start + np.argmin(bp_data[mid_start:mid_end])
                        
                        # 验证中间点确实是波谷
                        check_window = int(fs * 0.08)
                        check_start = max(mid_start, mid_valley_idx - check_window)
                        check_end = min(mid_end, mid_valley_idx + check_window)
                        
                        if bp_data[mid_valley_idx] <= np.min(bp_data[check_start:check_end]):
                            final_valleys.append(mid_valley_idx)
        
        refined_valleys = sorted(final_valleys)
    
    valley_indices = np.array(refined_valleys)
    logging.debug(f"Found {len(valley_indices)} BP valleys in {len(bp_data)} samples ({len(bp_data)/fs:.1f}s), HR: {estimated_hr:.0f} bpm")
    return valley_indices

def process_segment_data(subject_folder, segment, moved_data_dir=None, shift_amount=None):
    bp_path = os.path.join(subject_folder, f'{segment}_bp.csv')
    if not os.path.exists(bp_path):
        return None
    
    # Extract subject ID from folder path
    subject_id = os.path.basename(subject_folder)
    
    try:
        df_bp = pd.read_csv(bp_path, header=None, dtype=str)
        df_bp.columns = ['timestamp', 'bp']
        
        df_bp['timestamp'] = pd.to_numeric(df_bp['timestamp'], errors='coerce')
        df_bp['bp'] = pd.to_numeric(df_bp['bp'], errors='coerce')
        df_bp.dropna(inplace=True)
        
        if df_bp.empty:
            return None
            
        df_bp = df_bp.sort_values(by='timestamp').reset_index(drop=True)
        
        # Store original data for saving calibrated BP (before resampling)
        original_timestamps = df_bp['timestamp'].values
        original_bp = df_bp['bp'].values
        
        # Apply calibration to original data first (before resampling)
        original_bp_calibrated, applied_shift = apply_bp_calibration(original_bp, subject_id, segment, shift_amount)
        
        # Save calibrated BP data with original timestamps if moved_data_dir is provided
        if moved_data_dir is not None:
            save_calibrated_bp(original_bp_calibrated, original_timestamps, subject_id, segment, moved_data_dir)
        
        # Now do resampling for processing pipeline
        bp_resampled, bp_timestamps = resample_data(
            original_bp_calibrated,  # Use calibrated data for resampling
            original_timestamps,
            config.BP_SAMPLING_RATE,
            config.TARGET_SAMPLING_RATE
        )
        
        if len(bp_resampled) == 0:
            return None
        
        # Continue with the rest of processing using resampled data
        ppg_data_dict = {}
        available_sensors = []
        
        for sensor in config.SENSORS_TO_USE:
            sensor_path = os.path.join(subject_folder, f'{segment}_{sensor}.csv')
            if not os.path.exists(sensor_path):
                logging.debug(f"Warning: {sensor_path} not found. Skipping this sensor.")
                continue
            
            df_sensor = pd.read_csv(sensor_path)
            if 'ir' not in df_sensor.columns or 'timestamp' not in df_sensor.columns:
                logging.debug(f"Warning: Required columns missing in {sensor_path}")
                continue
                
            df_sensor['timestamp'] = pd.to_numeric(df_sensor['timestamp'], errors='coerce')
            df_sensor['ir'] = pd.to_numeric(df_sensor['ir'], errors='coerce')
            df_sensor.dropna(inplace=True)
            
            if df_sensor.empty:
                logging.debug(f"Warning: No valid data in {sensor_path}")
                continue
                
            df_sensor = df_sensor.sort_values(by='timestamp').reset_index(drop=True)
            
            try:
                ppg_filtered = butter_bandpass_filter(
                    df_sensor['ir'].values,
                    config.PPG_FILTER_LOW,
                    config.PPG_FILTER_HIGH,
                    config.SENSOR_SAMPLING_RATE
                )
                
                ppg_resampled, ppg_timestamps = resample_data(
                    ppg_filtered,
                    df_sensor['timestamp'].values,
                    config.SENSOR_SAMPLING_RATE,
                    config.TARGET_SAMPLING_RATE
                )
                
                if len(ppg_resampled) == 0:
                    logging.debug(f"Warning: Resampling failed for {sensor}")
                    continue
                    
                ppg_data_dict[sensor] = ppg_resampled 
                available_sensors.append(sensor)
                
            except Exception as e:
                logging.debug(f"Warning: Error processing {sensor}: {e}")
                continue
        
        if len(available_sensors) == 0:
            logging.debug(f"Warning: No valid sensors for {segment}")
            return None
        
        logging.debug(f"Available sensors for {segment}: {available_sensors} ({len(available_sensors)}/{len(config.SENSORS_TO_USE)})")
        
        min_length = min(len(bp_resampled), min(len(data) for data in ppg_data_dict.values()))
        
        if min_length < config.WINDOW_SIZE:
            logging.debug(f"Warning: Insufficient data length ({min_length}) for {segment}")
            return None
        
        bp_aligned = bp_resampled[:min_length]
        
        ppg_aligned = np.zeros((len(config.SENSORS_TO_USE), min_length))
        sensor_mask = np.zeros(len(config.SENSORS_TO_USE), dtype=bool)
        
        for i, sensor in enumerate(config.SENSORS_TO_USE):
            if sensor in ppg_data_dict:
                ppg_aligned[i, :] = ppg_data_dict[sensor][:min_length]
                sensor_mask[i] = True
        
        return ppg_aligned, bp_aligned, sensor_mask
        
    except Exception as e:
        logging.error(f"Error processing {segment}: {e}")
        return None

def create_slices_from_valleys(ppg_data, bp_data, sensor_mask, subject_folder, segment):
    """
    从血压波谷开始创建切片，每次滑动一个完整的血压波
    
    Args:
        ppg_data: PPG数据 (n_sensors, n_samples)
        bp_data: 血压数据 (n_samples,)
        sensor_mask: 传感器掩码
        subject_folder: Subject folder path (for loading HR)
        segment: Segment name (for loading HR)
    
    Returns:
        ppg_slices, bp_slices, mask_slices, window_start_indices: 列表，window_start_indices 为每个窗口在重采样轨上的起始样本索引
    """
    if ppg_data.shape[1] < config.WINDOW_SIZE:
        return [], [], [], []
    
    # Load heart rate if available
    avg_hr = load_heart_rate(subject_folder, segment)
    
    # 找到血压波谷
    valley_indices = find_bp_valleys(bp_data, config.TARGET_SAMPLING_RATE, avg_hr)
    
    if len(valley_indices) < 2:
        logging.debug("Warning: Insufficient BP valleys found, using original slicing method")
        ppg_slices, bp_slices, mask_slices = create_slices(ppg_data, bp_data, sensor_mask)
        # 退化路径没有精确的起始索引，这里使用等间隔步长近似（以 STRIDE_SIZE 为步长）
        window_start_indices = [i for i in range(0, bp_data.shape[0] - config.WINDOW_SIZE + 1, config.STRIDE_SIZE)] if ppg_slices else []
        return ppg_slices, bp_slices, mask_slices, window_start_indices
    
    ppg_slices, bp_slices, mask_slices = [], [], []
    window_start_indices = []
    
    # 从每个波谷开始创建10秒窗口
    for valley_idx in valley_indices:
        # 检查是否有足够的数据创建完整窗口
        if valley_idx + config.WINDOW_SIZE > ppg_data.shape[1]:
            break
            
        # 提取10秒窗口
        ppg_slice = ppg_data[:, valley_idx : valley_idx + config.WINDOW_SIZE]
        bp_slice = bp_data[valley_idx : valley_idx + config.WINDOW_SIZE]
        
        # 检查数据有效性
        if not np.any(np.isnan(ppg_slice)) and not np.any(np.isnan(bp_slice)):
            ppg_slices.append(ppg_slice)
            bp_slices.append(bp_slice)
            mask_slices.append(sensor_mask)
            window_start_indices.append(valley_idx)
    
    logging.debug(f"Created {len(ppg_slices)} valley-based slices from {len(valley_indices)} valleys")
    return ppg_slices, bp_slices, mask_slices, window_start_indices

def _compute_snr_per_channel(ppg_slice: np.ndarray, fs: int):
    """Compute per-channel SNR within [SNR_BAND_LOW, SNR_BAND_HIGH] using a simple spectral method.
    Steps: Welch/rFFT power; find f0 in HR range; signal band = [f0-δ, f0+δ]; noise band = band - signal; SNR = 10*log10(Psig / Pnoise_mean).
    Returns: np.ndarray shape (C,) with SNR in dB (float), NaN if cannot compute.
    """
    try:
        C, L = ppg_slice.shape
    except Exception:
        return None
    snrs = np.full((C,), np.nan, dtype=np.float32)
    # frequency grid
    freqs = np.fft.rfftfreq(L, d=1.0/fs)
    band_mask = (freqs >= getattr(config, 'SNR_BAND_LOW', 0.5)) & (freqs <= getattr(config, 'SNR_BAND_HIGH', 3.0))
    hr_low, hr_high = getattr(config, 'SNR_HR_RANGE', (0.7, 2.5))
    peak_nb = getattr(config, 'SNR_PEAK_NEIGHBOR', 0.12)
    eps = 1e-12
    for c in range(C):
        x = ppg_slice[c] - np.mean(ppg_slice[c])
        X = np.fft.rfft(x)
        P = (np.abs(X) ** 2)
        if not np.any(band_mask):
            continue
        # restrict to HR search range to find f0
        hr_mask = (freqs >= hr_low) & (freqs <= hr_high)
        if not np.any(hr_mask):
            continue
        hr_idx = np.argmax(P * hr_mask)
        f0 = freqs[hr_idx]
        sig_mask = (freqs >= max(getattr(config, 'SNR_BAND_LOW', 0.5), f0 - peak_nb)) & (freqs <= min(getattr(config, 'SNR_BAND_HIGH', 3.0), f0 + peak_nb))
        noise_mask = band_mask & (~sig_mask)
        Ps = np.sum(P[sig_mask])
        Pn = np.mean(P[noise_mask]) if np.any(noise_mask) else eps
        snr = 10.0 * np.log10((Ps + eps) / (Pn + eps))
        snrs[c] = np.float32(snr)
    return snrs

def create_slices(ppg_data, bp_data, sensor_mask):
    """原始的切片方法（保持向后兼容）"""
    if ppg_data.shape[1] < config.WINDOW_SIZE:
        return [], [], []
        
    ppg_slices, bp_slices, mask_slices = [], [], []

    for i in range(0, ppg_data.shape[1] - config.WINDOW_SIZE + 1, config.STRIDE_SIZE):
        ppg_slice = ppg_data[:, i : i + config.WINDOW_SIZE]
        bp_slice = bp_data[i : i + config.WINDOW_SIZE]
        
        if not np.any(np.isnan(ppg_slice)) and not np.any(np.isnan(bp_slice)):
            ppg_slices.append(ppg_slice)
            bp_slices.append(bp_slice)
            mask_slices.append(sensor_mask)
    
    return ppg_slices, bp_slices, mask_slices

def _estimate_hr_peak_and_peaks(x: np.ndarray, fs: int):
    """Estimate HR from peaks using scipy.find_peaks on a bandpassed-like signal x.
    Returns (hr_bpm, peak_indices, peak_count)."""
    try:
        if len(x) < max(8, int(0.5 * fs)):
            return 0.0, np.array([], dtype=int), 0
        # simple robust thresholds
        min_dist = int(0.3 * fs)
        sig = x - np.mean(x)
        std = np.std(sig) + 1e-6
        mean = np.mean(sig)
        candidates = [
            (mean + 0.2 * std, 0.1 * std),
            (mean + 0.1 * std, 0.05 * std),
            (mean, 0.02 * std),
        ]
        peaks = np.array([], dtype=int)
        for h, prom in candidates:
            peaks, _ = signal.find_peaks(sig, height=h, distance=min_dist, prominence=prom)
            if len(peaks) >= 3:
                break
        if len(peaks) < 2:
            return 0.0, peaks, len(peaks)
        peak_times = peaks / fs
        ibi_ms = np.diff(peak_times) * 1000.0
        valid = ibi_ms[(ibi_ms >= 300) & (ibi_ms <= 1200)]
        if len(valid) > 0:
            hr_bpm = float(np.mean(60000.0 / valid))
        else:
            hr_bpm = 0.0
        return hr_bpm, peaks, len(peaks)
    except Exception:
        return 0.0, np.array([], dtype=int), 0

def _estimate_hr_fft(x: np.ndarray, fs: int, min_bpm: int, max_bpm: int):
    """Estimate HR from FFT peak in [min_bpm, max_bpm] range."""
    try:
        if len(x) < max(8, int(0.5 * fs)):
            return 0.0
        sig = x - np.mean(x)
        P = np.abs(np.fft.rfft(sig)) ** 2
        freqs = np.fft.rfftfreq(len(sig), d=1.0/fs)
        mask = (freqs > (min_bpm/60.0)) & (freqs < (max_bpm/60.0))
        if not np.any(mask):
            return 0.0
        idx = np.argmax(P * mask)
        f0 = freqs[idx]
        return float(f0 * 60.0)
    except Exception:
        return 0.0

def _render_subject_overview(best_worst: dict, subject_id: str, split_name: str):
    """Render a per-subject overview figure with best and worst window per sensor.
    best_worst: dict[sensor_idx] -> {'best': cand, 'worst': cand}
    cand contains: signal (1D np.array), fs, peaks (idx array), hr_peak, hr_fft, hr_diff, hr_valid, snr_db, spr, window_sqi, sensor_name, segment
    """
    try:
        num_sensors = len(best_worst)
        if num_sensors == 0:
            return
        cols = 2
        rows = num_sensors
        fig, axes = plt.subplots(rows, cols, figsize=(cols*6, max(2*rows, 4)))
        if rows == 1:
            axes = np.array([axes])
        for i in range(num_sensors):
            row_axes = axes[i]
            # Best
            ax = row_axes[0]
            cand = best_worst[i].get('best', None)
            ax.set_title('')
            if cand is not None and isinstance(cand.get('signal', None), np.ndarray) and len(cand['signal']) > 0:
                sig = cand['signal']
                fs = cand.get('fs', 30)
                t = np.arange(len(sig)) / max(fs, 1)
                sig_plot = (sig - np.mean(sig)) / (np.std(sig) + 1e-6)
                ax.plot(t, sig_plot, color='#1f77b4', lw=1.0)
                peaks = cand.get('peaks', np.array([], dtype=int))
                if peaks is not None and len(peaks) > 0:
                    ax.scatter(peaks / max(fs, 1), sig_plot[peaks], c='#d62728', s=12, label='peaks')
                title = f"BEST - {cand.get('sensor_name','sensor')} ({cand.get('segment','')})\n"
                title += f"SNR {cand.get('snr_db', np.nan):.1f} dB, SQI {cand.get('window_sqi', np.nan):.3f}, HRp {cand.get('hr_peak',0):.0f}/HRf {cand.get('hr_fft',0):.0f}, d {cand.get('hr_diff', np.nan):.1f} bpm"
                ax.set_title(title, fontsize=9)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel(cand.get('sensor_name','sensor'))
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No BEST window', ha='center', va='center')
                ax.axis('off')
            # Worst
            ax = row_axes[1]
            cand = best_worst[i].get('worst', None)
            ax.set_title('')
            if cand is not None and isinstance(cand.get('signal', None), np.ndarray) and len(cand['signal']) > 0:
                sig = cand['signal']
                fs = cand.get('fs', 30)
                t = np.arange(len(sig)) / max(fs, 1)
                sig_plot = (sig - np.mean(sig)) / (np.std(sig) + 1e-6)
                ax.plot(t, sig_plot, color='#9467bd', lw=1.0)
                peaks = cand.get('peaks', np.array([], dtype=int))
                if peaks is not None and len(peaks) > 0:
                    ax.scatter(peaks / max(fs, 1), sig_plot[peaks], c='#ff7f0e', s=12, label='peaks')
                title = f"WORST - {cand.get('sensor_name','sensor')} ({cand.get('segment','')})\n"
                title += f"SNR {cand.get('snr_db', np.nan):.1f} dB, SQI {cand.get('window_sqi', np.nan):.3f}, HRp {cand.get('hr_peak',0):.0f}/HRf {cand.get('hr_fft',0):.0f}, d {cand.get('hr_diff', np.nan):.1f} bpm"
                ax.set_title(title, fontsize=9)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel(cand.get('sensor_name','sensor'))
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No WORST window', ha='center', va='center')
                ax.axis('off')
        plt.suptitle(f"Subject {subject_id} - {split_name} overview (Best vs Worst per Sensor)", fontsize=12)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        out_dir = getattr(config, 'SUBJECT_OVERVIEW_DIR', './')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{subject_id}_{split_name}_overview.png")
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
    except Exception as e:
        logging.debug(f"Failed to render subject overview for {subject_id}: {e}")


def _quality_score(cand: dict, strategy: str = None):
    """返回一个越大越好的分数；若缺少必要字段，返回-Inf。
    支持的策略：
    - 'pt'：-abs(hr_peak-true)
    - 'ft'：-abs(hr_fft-true)
    - 'min_true'：-min(abs(hr_peak-true), abs(hr_fft-true))（两者取更好者）
    - 'snr'：snr_db
    - 'pf_then_snr'：先用pf有效(1/0)+snr归一化
    - 'pt_then_snr'（默认）：若存在trueHR，用 -abs(hr_peak-true)；否则用 snr
    - 'composite'：0.6*snr - 0.4*hr_diff_peak_fft（越小越好）
    """
    if strategy is None:
        strategy = getattr(config, 'QUALITY_STRATEGY', 'pt_then_snr')
    snr = cand.get('snr_db', np.nan)
    hrp = cand.get('hr_peak', np.nan)
    hrf = cand.get('hr_fft', np.nan)
    tru = cand.get('true_hr', np.nan)
    pf = bool(cand.get('hr_valid', False))
    diff_pf = cand.get('hr_diff', np.nan)
    if strategy == 'pt':
        if np.isfinite(tru) and np.isfinite(hrp) and hrp > 0:
            return -abs(hrp - tru)
        return -np.inf
    if strategy == 'ft':
        if np.isfinite(tru) and np.isfinite(hrf) and hrf > 0:
            return -abs(hrf - tru)
        return -np.inf
    if strategy == 'min_true':
        vals = []
        if np.isfinite(tru) and np.isfinite(hrp) and hrp > 0:
            vals.append(abs(hrp - tru))
        if np.isfinite(tru) and np.isfinite(hrf) and hrf > 0:
            vals.append(abs(hrf - tru))
        return -min(vals) if vals else -np.inf
    if strategy == 'snr':
        return float(snr) if np.isfinite(snr) else -np.inf
    if strategy == 'pf_then_snr':
        base = 1.0 if pf else 0.0
        snr_v = float(snr) if np.isfinite(snr) else -100.0
        return base * 1000.0 + snr_v
    if strategy == 'pt_then_snr':
        if np.isfinite(tru) and np.isfinite(hrp) and hrp > 0:
            return -abs(hrp - tru)
        return float(snr) if np.isfinite(snr) else -np.inf
    if strategy == 'composite':
        snr_v = float(snr) if np.isfinite(snr) else -50.0
        diff_v = float(diff_pf) if np.isfinite(diff_pf) else 20.0
        return 0.6*snr_v - 0.4*diff_v
    # fallback
    return float(snr) if np.isfinite(snr) else -np.inf


def _update_best_worst(best_worst: dict, sensor_idx: int, cand: dict, strategy: str = None):
    """根据评分策略更新每个传感器的最佳/最差候选。"""
    try:
        score = _quality_score(cand, strategy)
        if not np.isfinite(score):
            return
        entry = best_worst.get(sensor_idx)
        if entry is None:
            best_worst[sensor_idx] = {'best': None, 'worst': None}
            entry = best_worst[sensor_idx]
        if entry['best'] is None or score > _quality_score(entry['best'], strategy):
            entry['best'] = cand
        if entry['worst'] is None or score < _quality_score(entry['worst'], strategy):
            entry['worst'] = cand
    except Exception:
        pass


def _select_best_worst(candidates: list, strategy: str = 'snr'):
    """选择最佳与最差窗口：
    - 最佳（best）：从合规层级最高的集合中选择（优先 strict>pf∧pt>pf>任意），按 strategy（默认SNR）最大。
    - 最差（worst）：从合规层级最低的集合中选择（优先 不达标<pf<pf∧pt<strict），按 strategy 最小。
    这样 worst 优先展示真正“不达标”的最差样本，而不是“合规集合中的最差”。
    """
    if not candidates:
        return {'best': None, 'worst': None}

    def level(c):
        strict_ok = bool(c.get('peaks_gt5', False)) and (not bool(c.get('is_flat', False))) and bool(c.get('pf_valid', False)) and bool(c.get('pt_valid', False))
        pf_pt_ok = bool(c.get('pf_valid', False)) and bool(c.get('pt_valid', False))
        pf_ok = bool(c.get('pf_valid', False))
        if strict_ok:
            return 3
        if pf_pt_ok:
            return 2
        if pf_ok:
            return 1
        return 0  # 不达标

    def score(c):
        if strategy == 'snr':
            v = c.get('snr_db', np.nan)
            return float(v) if np.isfinite(v) else -np.inf
        return _quality_score(c, strategy=strategy)

    # 计算每层集合
    by_level = {0: [], 1: [], 2: [], 3: []}
    for c in candidates:
        by_level[level(c)].append(c)

    # best 从最高非空层取分数最大
    best = None
    for lv in (3, 2, 1, 0):
        if by_level[lv]:
            best = max(by_level[lv], key=score)
            break

    # worst 从最低非空层取分数最小；若是非达标层（lv=0）且有平坦窗口，则优先在平坦中选 SNR 最低
    worst = None
    for lv in (0, 1, 2, 3):
        if by_level[lv]:
            if lv == 0:
                flats = [c for c in by_level[lv] if bool(c.get('is_flat', False))]
                if flats:
                    # 平坦优先，平坦中选 SNR 最低
                    def snr_value(c):
                        v = c.get('snr_db', np.nan)
                        return float(v) if np.isfinite(v) else np.inf
                    worst = min(flats, key=snr_value)
                else:
                    worst = min(by_level[lv], key=score)
            else:
                worst = min(by_level[lv], key=score)
            break

    return {'best': best, 'worst': worst}


def _process_one_subject(subject_id: str, split_name: str, moved_data_dir: str):
    """工作进程：处理一个受试者（两个segment），返回可聚合的数据与通过统计行。
    返回 dict: {
      'ppg': list[np.ndarray], 'bp': list[np.ndarray], 'mask': list[np.ndarray],
      'seg1_ppg': list, 'seg1_bp': list, 'seg1_mask': list,
      'seg7_ppg': list, 'seg7_bp': list, 'seg7_mask': list,
      'pass_rows': list[dict],
      'error': Optional[str]
    }
    同时函数内部负责写每人相关的CSV/PNG到唯一文件名，避免并发冲突。
    """
    try:
        subject_folder = os.path.join(config.RAW_DATA_DIR, subject_id)
        if not os.path.isdir(subject_folder):
            return {'error': f"subject folder missing: {subject_id}", 'subject_id': subject_id}

        # 每受试者共享的BP校准偏移
        shift_amount = calculate_bp_shift_amount(subject_id)

        # 聚合器
        all_ppg_for_subject, all_bp_for_subject, all_masks_for_subject = [], [], []
        seg1_ppg, seg1_bp, seg1_masks = [], [], []
        seg7_ppg, seg7_bp, seg7_masks = [], [], []
        pass_rows = []

        # 最佳/最差大图候选
        best_worst = {i: {'best': None, 'worst': None} for i in range(len(config.SENSORS_TO_USE))}
        strategy = getattr(config, 'QUALITY_STRATEGY', 'pt_then_snr')

        for segment in ['seg1', 'seg7']:
            result = process_segment_data(subject_folder, segment, moved_data_dir, shift_amount)
            if result is None:
                continue
            ppg_data, bp_data, sensor_mask = result
            ppg_slices, bp_slices, mask_slices, window_start_indices = create_slices_from_valleys(
                ppg_data, bp_data, sensor_mask, subject_folder, segment
            )
            df_hr_series = load_hr_series(subject_folder, segment)

            if not ppg_slices:
                continue

            # 添加到subject聚合器
            all_ppg_for_subject.extend(ppg_slices)
            all_bp_for_subject.extend(bp_slices)
            all_masks_for_subject.extend(mask_slices)
            if segment == 'seg1':
                seg1_ppg.extend(ppg_slices)
                seg1_bp.extend(bp_slices)
                seg1_masks.extend(mask_slices)
            else:
                seg7_ppg.extend(ppg_slices)
                seg7_bp.extend(bp_slices)
                seg7_masks.extend(mask_slices)

            # 计算窗口级统计、可视化和最佳/最差候选
            window_hr_valid = []
            window_all_valid = []
            sensor_signals = {}
            window_peaks = {s: {} for s in config.SENSORS_TO_USE}
            per_sensor_stats = {s: {'peak': [], 'fft': [], 'diff': []} for s in config.SENSORS_TO_USE}
            per_sensor_flags = {s: {
                'pf_valid_vec': [],
                'pt_valid_vec': [], 'pt_avail_vec': [],
                'ft_valid_vec': [], 'ft_avail_vec': [],
                'triple_valid_vec': [], 'triple_avail_vec': [],
                'pf_and_pt_vec': [],
                'strict_vec': [],  # peaks>5 & not flat & pf & pt
            } for s in config.SENSORS_TO_USE}
            fs = config.TARGET_SAMPLING_RATE
            window_size = config.WINDOW_SIZE
            stride = config.STRIDE_SIZE
            tol_bpm = getattr(config, 'HR_TOLERANCE_BPM', 5)

            per_window_rows = []
            # 为该段收集每通道候选后统一筛选（而不是逐行即刻更新），可更好满足严格条件
            segment_candidates = {i: [] for i in range(len(config.SENSORS_TO_USE))}

            for w, ppg_slice in enumerate(ppg_slices):
                C, L = ppg_slice.shape
                valid_flags = []
                # 窗口真实HR
                true_hr_window = np.nan
                if df_hr_series is not None:
                    start_idx = window_start_indices[w]
                    end_idx = start_idx + window_size
                    start_time = start_idx / fs
                    end_time = end_idx / fs
                    hr_in_win = df_hr_series[(df_hr_series['rel_time_s'] >= start_time) & (df_hr_series['rel_time_s'] < end_time)]['hr']
                    if not hr_in_win.empty:
                        true_hr_window = float(hr_in_win.mean())

                # SNR per channel
                try:
                    snr_vals = _compute_snr_per_channel(ppg_slice, fs=config.TARGET_SAMPLING_RATE)
                except Exception:
                    snr_vals = None

                for c in range(C):
                    x = ppg_slice[c]
                    hr_peak_bpm, peak_idx_list, peak_count = _estimate_hr_peak_and_peaks(x, fs)
                    hr_fft_bpm = _estimate_hr_fft(x, fs, getattr(config, 'HR_MIN_BPM', 50), getattr(config, 'HR_MAX_BPM', 200))
                    hr_diff = float(abs(hr_peak_bpm - hr_fft_bpm)) if (hr_peak_bpm > 0 and hr_fft_bpm > 0) else float('inf')
                    hr_valid = (hr_peak_bpm > 0 and hr_fft_bpm > 0 and hr_diff <= tol_bpm and peak_count >= getattr(config, 'PEAK_MIN_COUNT', 3))
                    valid_flags.append(hr_valid)

                    sensor = config.SENSORS_TO_USE[c]
                    window_peaks[sensor][w] = list(peak_idx_list)

                    if hr_peak_bpm > 0 and hr_fft_bpm > 0 and np.isfinite(hr_diff):
                        per_sensor_stats[sensor]['peak'].append(float(hr_peak_bpm))
                        per_sensor_stats[sensor]['fft'].append(float(hr_fft_bpm))
                        per_sensor_stats[sensor]['diff'].append(float(hr_diff))

                    snr_c = float(snr_vals[c]) if (snr_vals is not None and np.isfinite(snr_vals[c])) else np.nan
                    freqs = np.fft.rfftfreq(L, d=1.0/fs)
                    X = np.fft.rfft(x - np.mean(x))
                    P = np.abs(X) ** 2
                    band_mask = (freqs >= config.PPG_FILTER_LOW) & (freqs <= config.PPG_FILTER_HIGH)
                    band_power = float(P[band_mask].sum()) if np.any(band_mask) else np.nan
                    total_power = float(P.sum())
                    spr = float(np.clip(band_power / (total_power + 1e-8), 0, 1)) if np.isfinite(band_power) else np.nan
                    # 平坦检测：使用导数阈值占比 + 幅度范围
                    dx = np.diff(x)
                    flat_ratio = float(np.mean(np.abs(dx) < 1e-3))
                    amp_range = float(np.percentile(x, 95) - np.percentile(x, 5))
                    flat_thresh = getattr(config, 'FLAT_DX_RATIO_THRESH', 0.6)  # 超过该比例认为较平
                    amp_thresh = getattr(config, 'FLAT_AMP_MIN', 0.02)        # 幅度过小认为平
                    is_flat = (flat_ratio >= flat_thresh) or (amp_range <= amp_thresh)
                    peaks_gt5 = (peak_count > getattr(config, 'PEAK_MIN_FOR_BEST', 5))

                    # 一致性标记（在写行之前先算好）
                    pf_valid = bool(hr_valid)
                    pt_avail = (np.isfinite(true_hr_window) and hr_peak_bpm > 0)
                    pt_valid = (pt_avail and abs(hr_peak_bpm - true_hr_window) <= tol_bpm)
                    ft_avail = (np.isfinite(true_hr_window) and hr_fft_bpm > 0)
                    ft_valid = (ft_avail and abs(hr_fft_bpm - true_hr_window) <= tol_bpm)
                    triple_avail = (pt_avail and ft_avail and hr_peak_bpm > 0 and hr_fft_bpm > 0)
                    triple_valid = (triple_avail and pf_valid and pt_valid and ft_valid)
                    pf_and_pt = bool(pf_valid and pt_valid)
                    strict_ok = bool(peaks_gt5 and (not is_flat) and pf_valid and pt_valid)

                    start_idx = window_start_indices[w]
                    end_idx = start_idx + window_size
                    per_window_rows.append({
                        'split': split_name,
                        'subject_id': subject_id,
                        'segment': segment,
                        'window_idx': w,
                        'window_start_idx': int(start_idx),
                        'window_end_idx': int(end_idx),
                        'window_start_time_s': round(start_idx / fs, 6),
                        'window_end_time_s': round(end_idx / fs, 6),
                        'sensor_idx': int(c),
                        'sensor_name': sensor,
                        'peak_hr_bpm': float(hr_peak_bpm),
                        'fft_hr_bpm': float(hr_fft_bpm),
                        'hr_diff_bpm': float(hr_diff) if np.isfinite(hr_diff) else np.nan,
                        'peak_count': int(peak_count),
                        'snr_db': snr_c,
                        'spr': spr,
                        'band_power': band_power,
                        'total_power': total_power,
                        'hr_valid': bool(hr_valid),
                        'all_channels_valid': None,
                        'true_hr_bpm': float(true_hr_window) if np.isfinite(true_hr_window) else np.nan,
                        'peaks_gt5': bool(peaks_gt5),
                        'is_flat': bool(is_flat),
                        'pt_valid': bool(pt_valid),
                        'pf_and_pt': bool(pf_and_pt),
                        'strict': bool(strict_ok),
                    })

                    # 更新最佳/最差候选（使用原始或轻度滤波信号）
                    try:
                        x_f = butter_bandpass_filter(x, config.PPG_FILTER_LOW, config.PPG_FILTER_HIGH, fs, order=3)
                    except Exception:
                        x_f = x
                    # 将一致性标记写入统计向量
                    per_sensor_flags[sensor]['pf_valid_vec'].append(pf_valid)
                    per_sensor_flags[sensor]['pt_avail_vec'].append(bool(pt_avail))
                    per_sensor_flags[sensor]['pt_valid_vec'].append(bool(pt_valid))
                    per_sensor_flags[sensor]['ft_avail_vec'].append(bool(ft_avail))
                    per_sensor_flags[sensor]['ft_valid_vec'].append(bool(ft_valid))
                    per_sensor_flags[sensor]['triple_avail_vec'].append(bool(triple_avail))
                    per_sensor_flags[sensor]['triple_valid_vec'].append(bool(triple_valid))
                    per_sensor_flags[sensor]['pf_and_pt_vec'].append(bool(pf_and_pt))
                    per_sensor_flags[sensor]['strict_vec'].append(strict_ok)

                    cand = {
                        'signal': x_f.astype(float),
                        'fs': fs,
                        'peaks': peak_idx_list,
                        'hr_peak': float(hr_peak_bpm),
                        'hr_fft': float(hr_fft_bpm),
                        'hr_diff': float(hr_diff) if np.isfinite(hr_diff) else np.nan,
                        'hr_valid': bool(hr_valid),
                        'snr_db': snr_c,
                        'spr': spr,
                        'window_sqi': np.nan,  # 暂无统一SQI，后续可扩展
                        'sensor_idx': c,
                        'sensor_name': sensor,
                        'segment': segment,
                        'subject_id': subject_id,
                        'true_hr': float(true_hr_window) if np.isfinite(true_hr_window) else np.nan,
                        'peaks_gt5': bool(peaks_gt5),
                        'is_flat': bool(is_flat),
                        'pf_valid': bool(pf_valid),
                        'pt_valid': bool(pt_valid),
                        'window_start_idx': int(start_idx),
                        'window_end_idx': int(end_idx),
                    }
                    segment_candidates[c].append(cand)

                    if w == 0:
                        sensor_signals[sensor] = {'signal': ppg_data[c], 'fs': fs}

                all_flag = bool(all(valid_flags))
                start_row = len(per_window_rows) - C
                if start_row >= 0:
                    for i in range(C):
                        per_window_rows[start_row + i]['all_channels_valid'] = all_flag
                window_hr_valid.append(valid_flags)
                window_all_valid.append(all_flag)

            window_hr_valid = np.array(window_hr_valid)
            window_all_valid = np.array(window_all_valid)

            # 每段窗口通过统计行
            row = {
                'split': split_name,
                'subject_id': subject_id,
                'segment': segment,
                'total_windows': int(window_hr_valid.shape[0]),
                'all_channels_pass_windows': int(np.sum(window_all_valid)),
                'all_channels_pass_ratio': float(np.mean(window_all_valid)) if len(window_all_valid) > 0 else 0.0,
            }
            for s_idx, sensor in enumerate(config.SENSORS_TO_USE):
                row[f'{sensor}_pass_windows'] = int(np.sum(window_hr_valid[:, s_idx])) if window_hr_valid.size else 0
                row[f'{sensor}_pass_ratio'] = float(np.mean(window_hr_valid[:, s_idx])) if window_hr_valid.size else 0.0
            # 追加 ALL_SENSORS 的 pf_and_pt 与 strict 统计（窗口层）
            try:
                total_windows = int(window_hr_valid.shape[0])
                def _all_ok(flag_key, avail_key=None):
                    count = 0
                    for w_idx in range(total_windows):
                        ok = True
                        for s in config.SENSORS_TO_USE:
                            if avail_key is not None and not per_sensor_flags[s][avail_key][w_idx]:
                                ok = False; break
                            if not per_sensor_flags[s][flag_key][w_idx]:
                                ok = False; break
                        if ok:
                            count += 1
                    return count
                pf_and_pt_all = _all_ok('pf_and_pt_vec') if total_windows > 0 else 0
                strict_all = _all_ok('strict_vec') if total_windows > 0 else 0
                row['pf_and_pt_all_sensors_windows'] = pf_and_pt_all
                row['pf_and_pt_all_sensors_ratio'] = float(pf_and_pt_all / total_windows) if total_windows > 0 else 0.0
                row['strict_all_sensors_windows'] = strict_all
                row['strict_all_sensors_ratio'] = float(strict_all / total_windows) if total_windows > 0 else 0.0
            except Exception:
                pass
            pass_rows.append(row)

            # 在该分段结束后，根据候选统一选择最佳/最差（严格条件优先）并写入总览
            try:
                for c in range(len(config.SENSORS_TO_USE)):
                    sel = _select_best_worst(segment_candidates.get(c, []), strategy='snr')
                    if sel.get('best') is not None:
                        _update_best_worst(best_worst, c, sel['best'], strategy='snr')
                    if sel.get('worst') is not None:
                        _update_best_worst(best_worst, c, sel['worst'], strategy='snr')
            except Exception:
                pass

            # 可视化窗口矩阵
            try:
                _viz = globals().get('create_windowed_visualizations', None)
                if callable(_viz):
                    _viz(subject_id, f"{split_name}_{segment}", window_hr_valid, window_all_valid, sensor_signals, window_peaks, fs, window_size, stride, window_start_indices)
            except Exception:
                pass

            # 保存每段物理特征
            try:
                if per_window_rows:
                    out_dir = getattr(config, 'SUBJECT_OVERVIEW_DIR', './')
                    os.makedirs(out_dir, exist_ok=True)
                    cols = ['split','subject_id','segment','window_idx','window_start_idx','window_end_idx','window_start_time_s','window_end_time_s','sensor_idx','sensor_name','peak_hr_bpm','fft_hr_bpm','hr_diff_bpm','peak_count','snr_db','spr','band_power','total_power','hr_valid','all_channels_valid','true_hr_bpm','peaks_gt5','is_flat','pt_valid','pf_and_pt','strict']
                    per_window_df = pd.DataFrame(per_window_rows, columns=cols)
                    per_file = os.path.join(out_dir, f"physics_features_{subject_id}_{split_name}_{segment}.csv")
                    tmp_file = per_file + '.tmp'
                    per_window_df.to_csv(tmp_file, index=False)
                    os.replace(tmp_file, per_file)
            except Exception as e:
                logging.debug(f"保存 per-window 物理特征失败({subject_id},{segment}): {e}")

            # 保存统计表和矩阵
            try:
                out_dir = getattr(config, 'SUBJECT_OVERVIEW_DIR', './')
                os.makedirs(out_dir, exist_ok=True)
                total_windows = int(window_hr_valid.shape[0])

                per_sensor_rows = []
                for s_idx, sensor in enumerate(config.SENSORS_TO_USE):
                    valid_windows = int(np.sum(window_hr_valid[:, s_idx])) if total_windows > 0 else 0
                    ratio = float(valid_windows / total_windows) if total_windows > 0 else 0.0
                    def _mean_safe(arr):
                        return float(np.mean(arr)) if (arr is not None and len(arr) > 0) else np.nan
                    avg_peak = _mean_safe(per_sensor_stats[sensor]['peak'])
                    avg_fft = _mean_safe(per_sensor_stats[sensor]['fft'])
                    avg_diff = _mean_safe(per_sensor_stats[sensor]['diff'])
                    pf_valid_windows = int(np.sum(per_sensor_flags[sensor]['pf_valid_vec'])) if total_windows > 0 else 0
                    pf_valid_ratio = float(pf_valid_windows / total_windows) if total_windows > 0 else 0.0
                    pt_total = int(np.sum(per_sensor_flags[sensor]['pt_avail_vec']))
                    pt_valid_windows = int(np.sum(per_sensor_flags[sensor]['pt_valid_vec']))
                    pt_valid_ratio = float(pt_valid_windows / pt_total) if pt_total > 0 else np.nan
                    ft_total = int(np.sum(per_sensor_flags[sensor]['ft_avail_vec']))
                    ft_valid_windows = int(np.sum(per_sensor_flags[sensor]['ft_valid_vec']))
                    ft_valid_ratio = float(ft_valid_windows / ft_total) if ft_total > 0 else np.nan
                    triple_total = int(np.sum(per_sensor_flags[sensor]['triple_avail_vec']))
                    triple_valid_windows = int(np.sum(per_sensor_flags[sensor]['triple_valid_vec']))
                    triple_valid_ratio = float(triple_valid_windows / triple_total) if triple_total > 0 else np.nan
                    # 扩展统计：pf_and_pt 与 strict
                    pf_and_pt_windows = int(np.sum(per_sensor_flags[sensor]['pf_and_pt_vec'])) if total_windows > 0 else 0
                    pf_and_pt_ratio = float(pf_and_pt_windows / total_windows) if total_windows > 0 else 0.0
                    strict_windows = int(np.sum(per_sensor_flags[sensor]['strict_vec'])) if total_windows > 0 else 0
                    strict_ratio = float(strict_windows / total_windows) if total_windows > 0 else 0.0

                    per_sensor_rows.append({
                        'split': split_name,
                        'subject_id': subject_id,
                        'segment': segment,
                        'sensor': sensor,
                        'total_windows': total_windows,
                        'valid_windows': valid_windows,
                        'valid_ratio': round(ratio, 6),
                        'avg_peak_hr_bpm': round(avg_peak, 3) if np.isfinite(avg_peak) else np.nan,
                        'avg_fft_hr_bpm': round(avg_fft, 3) if np.isfinite(avg_fft) else np.nan,
                        'avg_hr_diff_bpm': round(avg_diff, 3) if np.isfinite(avg_diff) else np.nan,
                        'pf_valid_windows': pf_valid_windows,
                        'pf_valid_ratio': round(pf_valid_ratio, 6),
                        'pf_and_pt_windows': pf_and_pt_windows,
                        'pf_and_pt_ratio': round(pf_and_pt_ratio, 6),
                        'pt_total_truehr_windows': pt_total,
                        'pt_valid_windows': pt_valid_windows,
                        'pt_valid_ratio': round(pt_valid_ratio, 6) if np.isfinite(pt_valid_ratio) else np.nan,
                        'ft_total_truehr_windows': ft_total,
                        'ft_valid_windows': ft_valid_windows,
                        'ft_valid_ratio': round(ft_valid_ratio, 6) if np.isfinite(ft_valid_ratio) else np.nan,
                        'triple_total_truehr_windows': triple_total,
                        'triple_valid_windows': triple_valid_windows,
                        'triple_valid_ratio': round(triple_valid_ratio, 6) if np.isfinite(triple_valid_ratio) else np.nan,
                        'strict_windows': strict_windows,
                        'strict_ratio': round(strict_ratio, 6),
                    })

                def _all_sensors_count(flag_key, avail_key=None):
                    count = 0
                    for w_idx in range(total_windows):
                        ok = True
                        for s in config.SENSORS_TO_USE:
                            if avail_key is not None and not per_sensor_flags[s][avail_key][w_idx]:
                                ok = False; break
                            if not per_sensor_flags[s][flag_key][w_idx]:
                                ok = False; break
                        if ok:
                            count += 1
                    return count

                all_valid_count = int(np.sum(window_all_valid)) if total_windows > 0 else 0
                all_valid_ratio = float(all_valid_count / total_windows) if total_windows > 0 else 0.0
                pt_all_count = _all_sensors_count('pt_valid_vec', 'pt_avail_vec') if total_windows > 0 else 0
                ft_all_count = _all_sensors_count('ft_valid_vec', 'ft_avail_vec') if total_windows > 0 else 0
                triple_all_count = _all_sensors_count('triple_valid_vec', 'triple_avail_vec') if total_windows > 0 else 0
                pf_and_pt_all_count = _all_sensors_count('pf_and_pt_vec') if total_windows > 0 else 0
                strict_all_count = _all_sensors_count('strict_vec') if total_windows > 0 else 0
                pt_all_ratio = float(pt_all_count / total_windows) if total_windows > 0 else 0.0
                ft_all_ratio = float(ft_all_count / total_windows) if total_windows > 0 else 0.0
                triple_all_ratio = float(triple_all_count / total_windows) if total_windows > 0 else 0.0
                pf_and_pt_all_ratio = float(pf_and_pt_all_count / total_windows) if total_windows > 0 else 0.0
                strict_all_ratio = float(strict_all_count / total_windows) if total_windows > 0 else 0.0

                per_sensor_rows.append({
                    'split': split_name,
                    'subject_id': subject_id,
                    'segment': segment,
                    'sensor': 'ALL_SENSORS',
                    'total_windows': total_windows,
                    'valid_windows': all_valid_count,
                    'valid_ratio': round(all_valid_ratio, 6),
                    'pf_valid_windows': all_valid_count,
                    'pf_valid_ratio': round(all_valid_ratio, 6),
                    'pf_and_pt_all_sensors_windows': pf_and_pt_all_count,
                    'pf_and_pt_all_sensors_ratio': round(pf_and_pt_all_ratio, 6),
                    'pt_all_sensors_windows': pt_all_count,
                    'pt_all_sensors_ratio': round(pt_all_ratio, 6),
                    'ft_all_sensors_windows': ft_all_count,
                    'ft_all_sensors_ratio': round(ft_all_ratio, 6),
                    'triple_all_sensors_windows': triple_all_count,
                    'triple_all_sensors_ratio': round(triple_all_ratio, 6),
                    'strict_all_sensors_windows': strict_all_count,
                    'strict_all_sensors_ratio': round(strict_all_ratio, 6),
                })

                per_sensor_df = pd.DataFrame(per_sensor_rows)
                stats_file = os.path.join(out_dir, f"window_stats_{subject_id}_{split_name}_{segment}.csv")
                tmp_stats = stats_file + '.tmp'
                per_sensor_df.to_csv(tmp_stats, index=False)
                os.replace(tmp_stats, stats_file)

                # 详情与矩阵
                matrix_rows = []
                for w_idx in range(total_windows):
                    rowm = {
                        'split': split_name,
                        'subject_id': subject_id,
                        'segment': segment,
                        'window_idx': w_idx,
                        'all_sensors_valid': bool(window_all_valid[w_idx]) if total_windows > 0 else False,
                    }
                    for s_idx, sensor in enumerate(config.SENSORS_TO_USE):
                        rowm[f"valid_{sensor}"] = bool(window_hr_valid[w_idx, s_idx]) if total_windows > 0 else False
                    matrix_rows.append(rowm)
                matrix_df = pd.DataFrame(matrix_rows)
                matrix_file = os.path.join(out_dir, f"window_hr_valid_matrix_{subject_id}_{split_name}_{segment}.csv")
                tmp_mat = matrix_file + '.tmp'
                matrix_df.to_csv(tmp_mat, index=False)
                os.replace(tmp_mat, matrix_file)
            except Exception as e:
                logging.debug(f"保存窗口统计失败({subject_id},{segment}): {e}")

        # 受试者处理完成后绘制最佳/最差总览（默认启用）
        if getattr(config, 'SQI_SUBJECT_OVERVIEW', True):
            try:
                _render_subject_overview(best_worst, subject_id=subject_id, split_name=split_name)
            except Exception:
                pass

        return {
            'subject_id': subject_id,
            'ppg': all_ppg_for_subject,
            'bp': all_bp_for_subject,
            'mask': all_masks_for_subject,
            'seg1_ppg': seg1_ppg,
            'seg1_bp': seg1_bp,
            'seg1_mask': seg1_masks,
            'seg7_ppg': seg7_ppg,
            'seg7_bp': seg7_bp,
            'seg7_mask': seg7_masks,
            'pass_rows': pass_rows,
        }
    except Exception as e:
        return {'error': str(e), 'subject_id': subject_id}

def main():
    decompress_archives(config.SHARED_DATA_DIR)
    
    all_subjects = sorted(list(set(config.TRAIN_SUBJECTS + config.VALID_SUBJECTS + config.TEST_SUBJECTS)))
    organize_raw_data(config.SHARED_DATA_DIR, config.RAW_DATA_DIR, all_subjects)

    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    
    # Create moved data directory for saving calibrated BP data
    moved_data_dir = getattr(config, 'MOVED_DATA_DIR', os.path.join(os.path.dirname(config.PROCESSED_DATA_DIR), 'moved'))
    os.makedirs(moved_data_dir, exist_ok=True)
    logging.debug(f"Created moved data directory: {moved_data_dir}")
    
    subject_splits = {"train": config.TRAIN_SUBJECTS, "validation": config.VALID_SUBJECTS, "test": config.TEST_SUBJECTS}
    
    for split_name, subjects in subject_splits.items():
        if not subjects:
            continue
        
        # --- 并行执行每个受试者 ---
        logging.debug(f"Generating {split_name.upper()} SET")
        all_ppg_for_split, all_bp_for_split, all_masks_for_split = [], [], []
        seg1_ppg_for_split, seg1_bp_for_split, seg1_masks_for_split = [], [], []
        seg7_ppg_for_split, seg7_bp_for_split, seg7_masks_for_split = [], [], []
        split_subject_pass_rows = []

        max_workers = int(getattr(config, 'NUM_WORKERS', 8) or 8)
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_process_one_subject, subject_id, split_name, moved_data_dir): subject_id for subject_id in subjects}
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {split_name} subjects (parallel)"):
                sid = futures[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    logging.debug(f"Worker crashed for {sid}: {e}")
                    continue
                if res is None:
                    continue
                if res.get('error'):
                    logging.debug(f"Subject {sid} error: {res['error']}")
                    continue
                # 聚合
                all_ppg_for_split.extend(res.get('ppg', []))
                all_bp_for_split.extend(res.get('bp', []))
                all_masks_for_split.extend(res.get('mask', []))
                seg1_ppg_for_split.extend(res.get('seg1_ppg', []))
                seg1_bp_for_split.extend(res.get('seg1_bp', []))
                seg1_masks_for_split.extend(res.get('seg1_mask', []))
                seg7_ppg_for_split.extend(res.get('seg7_ppg', []))
                seg7_bp_for_split.extend(res.get('seg7_bp', []))
                seg7_masks_for_split.extend(res.get('seg7_mask', []))
                split_subject_pass_rows.extend(res.get('pass_rows', []))

        # 在当前 split 结束后，保存“所有人的通过统计”
        try:
            if split_subject_pass_rows:
                out_dir = getattr(config, 'SQI_REPORT_DIR', './')
                os.makedirs(out_dir, exist_ok=True)
                pass_df = pd.DataFrame(split_subject_pass_rows)
                pass_file = os.path.join(out_dir, f'subject_pass_summary_{split_name}.csv')
                tmp_pass = pass_file + '.tmp'
                pass_df.to_csv(tmp_pass, index=False)
                os.replace(tmp_pass, pass_file)
                print(f"💾 保存所有人的通过统计: {pass_file}")
        except Exception as e:
            logging.debug(f"保存所有人通过统计失败: {e}")

        # 保存split级数据
        if not all_ppg_for_split:
            logging.debug(f"No data generated for {split_name} set.")
            continue

        all_ppg_np = np.array(all_ppg_for_split, dtype=np.float32)
        all_bp_np = np.array(all_bp_for_split, dtype=np.float32)
        all_masks_np = np.array(all_masks_for_split, dtype=bool)

        logging.debug(f"Generated {len(all_ppg_np)} samples for {split_name} set (combined)")
        logging.debug(f"PPG shape: {all_ppg_np.shape}, BP shape: {all_bp_np.shape}, Mask shape: {all_masks_np.shape}")

        # 统计传感器可用率
        try:
            sensor_availability = np.mean(all_masks_np, axis=0)
            for i, sensor in enumerate(config.SENSORS_TO_USE):
                logging.debug(f"  {sensor} availability: {sensor_availability[i]*100:.1f}%")
        except Exception:
            pass

        save_path_npz = os.path.join(config.PROCESSED_DATA_DIR, f"{split_name}_data.npz")
        np.savez(save_path_npz, ppg=all_ppg_np, bp=all_bp_np, sensor_mask=all_masks_np)
        logging.debug(f"Saved combined dataset to {save_path_npz}")

        # Save seg1-only dataset
        if seg1_ppg_for_split:
            seg1_ppg_np = np.array(seg1_ppg_for_split, dtype=np.float32)
            seg1_bp_np = np.array(seg1_bp_for_split, dtype=np.float32)
            seg1_masks_np = np.array(seg1_masks_for_split, dtype=bool)
            seg1_save_path = os.path.join(config.PROCESSED_DATA_DIR, f"{split_name}_seg1_data.npz")
            np.savez(seg1_save_path, ppg=seg1_ppg_np, bp=seg1_bp_np, sensor_mask=seg1_masks_np)
            logging.debug(f"Saved seg1 dataset to {seg1_save_path}")

        # Save seg7-only dataset
        if seg7_ppg_for_split:
            seg7_ppg_np = np.array(seg7_ppg_for_split, dtype=np.float32)
            seg7_bp_np = np.array(seg7_bp_for_split, dtype=np.float32)
            seg7_masks_np = np.array(seg7_masks_for_split, dtype=bool)
            seg7_save_path = os.path.join(config.PROCESSED_DATA_DIR, f"{split_name}_seg7_data.npz")
            np.savez(seg7_save_path, ppg=seg7_ppg_np, bp=seg7_bp_np, sensor_mask=seg7_masks_np)
            logging.debug(f"Saved seg7 dataset to {seg7_save_path}")

        # 保存少量CSV样本
        try:
            NUM_SAMPLES_TO_SAVE = 5
            csv_sample_dir = os.path.join(config.RESULTS_DIR, f"{split_name}_csv_samples")
            os.makedirs(csv_sample_dir, exist_ok=True)
            num_total_samples = len(all_ppg_np)
            sample_indices = np.random.choice(num_total_samples, size=min(NUM_SAMPLES_TO_SAVE, num_total_samples), replace=False)
            for i, idx in enumerate(sample_indices):
                ppg_sample = all_ppg_np[idx]
                mask_sample = all_masks_np[idx]
                ppg_df = pd.DataFrame(ppg_sample.T, columns=[f'{s}_ir' for s in config.SENSORS_TO_USE])
                ppg_df.to_csv(os.path.join(csv_sample_dir, f"sample_{i}_ppg.csv"), index=False)
                mask_df = pd.DataFrame({
                    'sensor': config.SENSORS_TO_USE,
                    'available': mask_sample,
                    'active_sensors': [s if mask_sample[j] else 'MISSING' for j, s in enumerate(config.SENSORS_TO_USE)]
                })
                mask_df.to_csv(os.path.join(csv_sample_dir, f"sample_{i}_mask.csv"), index=False)
                bp_sample = all_bp_np[idx]
                bp_df = pd.DataFrame(bp_sample, columns=['bp'])
                bp_df.to_csv(os.path.join(csv_sample_dir, f"sample_{i}_bp.csv"), index=False)
        except Exception:
            pass
    
    logging.debug(f"All calibrated BP data saved to: {moved_data_dir}")
    print(f"\nCalibrated BP data saved to: {moved_data_dir}")
    print("Directory structure:")
    print("moved/")
    for subject_id in all_subjects:
        moved_subject_dir = os.path.join(moved_data_dir, subject_id)
        if os.path.exists(moved_subject_dir):
            files = os.listdir(moved_subject_dir)
            bp_files = [f for f in files if f.endswith('_bp.csv')]
            if bp_files:
                print(f"  {subject_id}/")
                for bp_file in sorted(bp_files):
                    print(f"    {bp_file}")
    
    # Print summary of generated datasets
    print(f"\nGenerated datasets in: {config.PROCESSED_DATA_DIR}")
    print("Dataset files:")
    for split_name in ["train", "validation", "test"]:
        # Combined dataset
        combined_path = os.path.join(config.PROCESSED_DATA_DIR, f"{split_name}_data.npz")
        if os.path.exists(combined_path):
            print(f"  {split_name}_data.npz (seg1 + seg7 combined)")
        
        # Seg1 only dataset
        seg1_path = os.path.join(config.PROCESSED_DATA_DIR, f"{split_name}_seg1_data.npz")
        if os.path.exists(seg1_path):
            print(f"  {split_name}_seg1_data.npz (seg1 only)")
        
        # Seg7 only dataset
        seg7_path = os.path.join(config.PROCESSED_DATA_DIR, f"{split_name}_seg7_data.npz")
        if os.path.exists(seg7_path):
            print(f"  {split_name}_seg7_data.npz (seg7 only)")

    # SQI 报告在并行改造后默认关闭；如需保留，可在 worker 级别累积并返回。

if __name__ == "__main__":
    main()