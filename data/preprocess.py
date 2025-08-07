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

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   filename='processing.log',
                   filemode='w')

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

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
            dbp_mean = df_dbp['dbp'].mean() if not df_dbp.empty else None
        else:
            logging.debug(f"Diastolic BP file not found: {diastolic_path}")
            dbp_mean = None
            
        if sbp_mean is not None and dbp_mean is not None:
            logging.debug(f"Biopac BP values for {segment}: SBP={sbp_mean:.1f}, DBP={dbp_mean:.1f}")
        
        return sbp_mean, dbp_mean
        
    except Exception as e:
        logging.error(f"Error loading Biopac BP values: {e}")
        return None, None

def apply_bp_calibration(bp_data, subject_id, segment):
    """
    Apply BP calibration based on Omron and Biopac differences
    
    Args:
        bp_data: BP waveform data
        subject_id: Subject ID (e.g., '00042')
        segment: Segment name (e.g., 'seg1')
    
    Returns:
        Calibrated BP data and shift amount (for printing)
    """
    # Load all Omron data
    omron_data = load_omron_data()
    
    # Look up this subject and segment
    omron_key = f"{subject_id}_{segment}"
    if omron_key not in omron_data:
        logging.debug(f"No Omron data for {subject_id} {segment}")
        return bp_data, None
    
    omron_sbp, omron_dbp = omron_data[omron_key]
    
    # Load Biopac SBP/DBP
    subject_folder = os.path.join(config.RAW_DATA_DIR, subject_id)
    biopac_sbp, biopac_dbp = load_biopac_bp_values(subject_folder, segment)
    
    if biopac_sbp is None or biopac_dbp is None:
        logging.debug(f"No Biopac SBP/DBP data for {subject_id} {segment}")
        return bp_data, None
    
    # Calculate differences
    sbp_diff = omron_sbp - biopac_sbp
    dbp_diff = omron_dbp - biopac_dbp
    avg_diff = (sbp_diff + dbp_diff) / 2
    
    # Apply shift if difference is reasonable (within ±10 mmHg)
    if abs(sbp_diff - dbp_diff) <= 10:
        calibrated_bp = bp_data + avg_diff
        print(f"segment {segment}: BP waveform shifted by {avg_diff:+.1f} mmHg")
        return calibrated_bp, avg_diff
    else:
        print(f"segment {segment}: No shift applied, delta too large ({sbp_diff - dbp_diff:.1f} mmHg)")
        return bp_data, None

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

def process_segment_data(subject_folder, segment):
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
        
        bp_resampled, bp_timestamps = resample_data(
            df_bp['bp'].values,
            df_bp['timestamp'].values,
            config.BP_SAMPLING_RATE,
            config.TARGET_SAMPLING_RATE
        )
        
        if len(bp_resampled) == 0:
            return None
        
        # Apply BP calibration - now passing subject_id
        bp_calibrated, shift_amount = apply_bp_calibration(bp_resampled, subject_id, segment)
        
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
        
        min_length = min(len(bp_calibrated), min(len(data) for data in ppg_data_dict.values()))
        
        if min_length < config.WINDOW_SIZE:
            logging.debug(f"Warning: Insufficient data length ({min_length}) for {segment}")
            return None
        
        bp_aligned = bp_calibrated[:min_length]
        
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
        ppg_slices, bp_slices, mask_slices: 切片列表
    """
    if ppg_data.shape[1] < config.WINDOW_SIZE:
        return [], [], []
    
    # Load heart rate if available
    avg_hr = load_heart_rate(subject_folder, segment)
    
    # 找到血压波谷
    valley_indices = find_bp_valleys(bp_data, config.TARGET_SAMPLING_RATE, avg_hr)
    
    if len(valley_indices) < 2:
        logging.debug("Warning: Insufficient BP valleys found, using original slicing method")
        return create_slices(ppg_data, bp_data, sensor_mask)
    
    ppg_slices, bp_slices, mask_slices = [], [], []
    
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
    
    logging.debug(f"Created {len(ppg_slices)} valley-based slices from {len(valley_indices)} valleys")
    return ppg_slices, bp_slices, mask_slices

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

def main():
    decompress_archives(config.SHARED_DATA_DIR)
    
    all_subjects = sorted(list(set(config.TRAIN_SUBJECTS + config.VALID_SUBJECTS + config.TEST_SUBJECTS)))
    organize_raw_data(config.SHARED_DATA_DIR, config.RAW_DATA_DIR, all_subjects)

    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    subject_splits = {"train": config.TRAIN_SUBJECTS, "validation": config.VALID_SUBJECTS, "test": config.TEST_SUBJECTS}
    
    for split_name, subjects in subject_splits.items():
        if not subjects:
            continue
            
        all_ppg_for_split, all_bp_for_split, all_masks_for_split = [], [], []
        
        logging.debug(f"Generating {split_name.upper()} SET")
        for subject_id in tqdm(subjects, desc=f"Processing {split_name} subjects"):
            subject_folder = os.path.join(config.RAW_DATA_DIR, subject_id)
            if not os.path.isdir(subject_folder):
                logging.warning(f"Warning: Directory for subject {subject_id} not found.")
                continue

            for segment in ['seg1', 'seg7']:
                logging.debug(f"Processing {subject_id} - {segment}")
                result = process_segment_data(subject_folder, segment)
                
                if result is not None:
                    ppg_data, bp_data, sensor_mask = result
                    # 使用新的从波谷开始的切片方法，传入subject_folder和segment用于加载HR
                    ppg_slices, bp_slices, mask_slices = create_slices_from_valleys(
                        ppg_data, bp_data, sensor_mask, subject_folder, segment
                    )
                    
                    if ppg_slices:
                        all_ppg_for_split.extend(ppg_slices)
                        all_bp_for_split.extend(bp_slices)
                        all_masks_for_split.extend(mask_slices)
                        logging.debug(f"Generated {len(ppg_slices)} valley-based slices from {segment}")
                    else:
                        logging.debug(f"No valid slices from {segment}")
                else:
                    logging.debug(f"No data processed for {segment}")

        if not all_ppg_for_split:
            logging.debug(f"No data generated for {split_name} set.")
            continue

        all_ppg_np = np.array(all_ppg_for_split, dtype=np.float32)
        all_bp_np = np.array(all_bp_for_split, dtype=np.float32)
        all_masks_np = np.array(all_masks_for_split, dtype=bool)
        
        logging.debug(f"Generated {len(all_ppg_np)} samples for {split_name} set")
        logging.debug(f"PPG shape: {all_ppg_np.shape}, BP shape: {all_bp_np.shape}, Mask shape: {all_masks_np.shape}")
        
        # 统计传感器可用率
        sensor_availability = np.mean(all_masks_np, axis=0)
        for i, sensor in enumerate(config.SENSORS_TO_USE):
            logging.debug(f"  {sensor} availability: {sensor_availability[i]*100:.1f}%")
        
        save_path_npz = os.path.join(config.PROCESSED_DATA_DIR, f"{split_name}_data.npz")
        np.savez(save_path_npz, ppg=all_ppg_np, bp=all_bp_np, sensor_mask=all_masks_np)
        logging.debug(f"Saved to {save_path_npz}")

        NUM_SAMPLES_TO_SAVE = 5
        csv_sample_dir = os.path.join(config.PROCESSED_DATA_DIR, f"{split_name}_csv_samples")
        os.makedirs(csv_sample_dir, exist_ok=True)
        
        num_total_samples = len(all_ppg_np)
        sample_indices = np.random.choice(num_total_samples, size=min(NUM_SAMPLES_TO_SAVE, num_total_samples), replace=False)
        
        for i, idx in enumerate(sample_indices):
            ppg_sample = all_ppg_np[idx]
            mask_sample = all_masks_np[idx]
            
            # 只保存有效传感器的数据
            active_sensors = [config.SENSORS_TO_USE[j] for j in range(len(config.SENSORS_TO_USE)) if mask_sample[j]]
            ppg_df = pd.DataFrame(ppg_sample.T, columns=[f'{s}_ir' for s in config.SENSORS_TO_USE])
            ppg_df.to_csv(os.path.join(csv_sample_dir, f"sample_{i}_ppg.csv"), index=False)
            
            # 保存传感器掩码信息
            mask_df = pd.DataFrame({
                'sensor': config.SENSORS_TO_USE,
                'available': mask_sample,
                'active_sensors': [s if mask_sample[j] else 'MISSING' for j, s in enumerate(config.SENSORS_TO_USE)]
            })
            mask_df.to_csv(os.path.join(csv_sample_dir, f"sample_{i}_mask.csv"), index=False)
            
            bp_sample = all_bp_np[idx]
            bp_df = pd.DataFrame(bp_sample, columns=['bp'])
            bp_df.to_csv(os.path.join(csv_sample_dir, f"sample_{i}_bp.csv"), index=False)

if __name__ == "__main__":
    main()