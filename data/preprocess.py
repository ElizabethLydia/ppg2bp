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

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
import config

def decompress_archives(source_dir):
    print(f"--- Scanning for archives in: {source_dir} ---")
    archives = glob.glob(os.path.join(source_dir, "*.tar*"))
    if not archives:
        print("No archives found to decompress.")
        return
    for archive_path in archives:
        archive_name = os.path.basename(archive_path)
        extracted_folder_name = archive_name.split('.')[0]
        extracted_folder_path = os.path.join(source_dir, extracted_folder_name)
        if os.path.isdir(extracted_folder_path):
            print(f"Directory '{extracted_folder_name}' already exists. Skipping.")
            continue
        print(f"Decompressing '{archive_name}'...")
        try:
            with tarfile.open(archive_path, "r:*") as tar:
                tar.extractall(path=source_dir)
            print("Successfully decompressed.")
        except Exception as e:
            print(f"Error decompressing {archive_name}: {e}")
    print("--- Archive decompression finished ---")

def organize_raw_data(source_base_dir, dest_dir, subjects_to_process):
    print(f"--- Organizing raw data from: {source_base_dir} ---")
    if not os.path.isdir(source_base_dir):
        print(f"Error: Source directory not found at {source_base_dir}")
        return
    task_mapping = {'1': 'seg1', '7': 'seg7'}
    for subject_id in tqdm(subjects_to_process, desc="Organizing subjects"):
        subject_dest_folder = os.path.join(dest_dir, subject_id)
        os.makedirs(subject_dest_folder, exist_ok=True)
        subject_paths = glob.glob(os.path.join(source_base_dir, "*", subject_id))
        if not subject_paths:
            print(f"Warning: No folder found for subject {subject_id}")
            continue
        subject_path = subject_paths[0]
        for task_id, seg_name in task_mapping.items():
            task_folder = os.path.join(subject_path, task_id)
            if not os.path.isdir(task_folder): continue
            biopac_folder = os.path.join(task_folder, "Biopac")
            bp_src_path = os.path.join(biopac_folder, "bp.csv")
            if os.path.exists(bp_src_path):
                shutil.copy2(bp_src_path, os.path.join(subject_dest_folder, f"{seg_name}_bp.csv"))
            hub_folder = os.path.join(task_folder, "HUB")
            if os.path.isdir(hub_folder):
                for sensor_src_path in glob.glob(os.path.join(hub_folder, "*.csv")):
                    shutil.copy2(sensor_src_path, os.path.join(subject_dest_folder, f"{seg_name}_{os.path.basename(sensor_src_path)}"))
    print("--- Raw data organization finished ---")

def get_bp_correction_offset_from_files(subject_id, segment, source_base_dir):
    try:
        #读取Omron数据
        df_omron = pd.read_csv(config.OMRON_CSV_PATH)
        subject_omron_data = df_omron[df_omron['id'] == int(subject_id)]
        if subject_omron_data.empty or subject_omron_data[['sbp', 'dbp']].isnull().values.any():
            return 0.0
        omron_sbp = subject_omron_data.iloc[0]['sbp']
        omron_dbp = subject_omron_data.iloc[0]['dbp']
        

        task_id = '1' if segment == 'seg1' else '7'
        subject_paths = glob.glob(os.path.join(source_base_dir, "*", subject_id))
        if not subject_paths:
            print(f"  - Warning: No folder found for subject {subject_id}")
            return 0.0
        
        subject_path = subject_paths[0]
        biopac_folder = os.path.join(subject_path, task_id, "Biopac")
        
        dbp_path = os.path.join(biopac_folder, "diastolic_bp.csv")
        sbp_path = os.path.join(biopac_folder, "systolic_bp.csv")
        
        if not os.path.exists(dbp_path) or not os.path.exists(sbp_path):
            print(f"  - Warning: DBP or SBP file not found for {subject_id}-{segment}")
            return 0.0
        
        # 读取DBP数据
        df_dbp = pd.read_csv(dbp_path)
        if 'timestamp' not in df_dbp.columns:
            df_dbp = pd.read_csv(dbp_path, header=None)
            df_dbp.columns = ['timestamp', 'dbp']
        df_dbp['dbp'] = pd.to_numeric(df_dbp.iloc[:, 1], errors='coerce')
        df_dbp.dropna(inplace=True)
        
        # 读取SBP数据
        df_sbp = pd.read_csv(sbp_path)
        if 'timestamp' not in df_sbp.columns:
            df_sbp = pd.read_csv(sbp_path, header=None)
            df_sbp.columns = ['timestamp', 'sbp']
        df_sbp['sbp'] = pd.to_numeric(df_sbp.iloc[:, 1], errors='coerce')
        df_sbp.dropna(inplace=True)
        
        # 计算平均值
        biopac_dbp_mean = df_dbp['dbp'].mean()
        biopac_sbp_mean = df_sbp['sbp'].mean()
        
        # 计算脉压差
        pp_omron = omron_sbp - omron_dbp
        pp_biopac = biopac_sbp_mean - biopac_dbp_mean
        pp_diff = abs(pp_omron - pp_biopac)
        
        print(f"  - Pulse Pressure Check: Omron_PP={pp_omron:.2f}, Biopac_PP={pp_biopac:.2f}, Difference={pp_diff:.2f}")
        print(f"  - Biopac SBP mean: {biopac_sbp_mean:.2f}, DBP mean: {biopac_dbp_mean:.2f}")
        
        if pp_diff <= config.PULSE_PRESSURE_DIFF_THRESHOLD:
            offset = ((omron_sbp - biopac_sbp_mean) + (omron_dbp - biopac_dbp_mean)) / 2.0
            print(f"  - SUCCESS: Applying offset of {offset:.2f} mmHg.")
            return offset
        else:
            print(f"  - Warning: Pulse pressure difference exceeds threshold. No correction.")
            return 0.0
            
    except Exception as e:
        print(f"  - Warning: An error occurred during BP correction for {subject_id}: {e}")
        return 0.0

def butter_bandpass_filter_stable(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    if low <= 0 or high >= 1 or low >= high:
        print(f"  - Warning: Invalid filter parameters (low={low:.4f}, high={high:.4f}). Returning original data.")
        return data
    
    if np.all(np.isnan(data)):
        print("  - Warning: All input data is NaN. Returning as is.")
        return data
    
    try:
        sos = signal.butter(order, [low, high], btype='band', output='sos')
        filtered = signal.sosfiltfilt(sos, data)
        
        if np.all(np.isnan(filtered)):
            print("  - Warning: Filter produced all NaN values. Trying lower order...")
            sos = signal.butter(2, [low, high], btype='band', output='sos')
            filtered = signal.sosfiltfilt(sos, data)
            
            if np.all(np.isnan(filtered)):
                print("  - Warning: Filter still producing NaN. Returning original data.")
                return data
        
        return filtered
        
    except Exception as e:
        print(f"  - Warning: Filter failed with error: {e}. Returning original data.")
        return data

def resample_data(data, timestamps, original_fs, target_fs):
    if len(data) < 2: return np.array([]), np.array([])
    
    if not np.all(np.diff(timestamps) > 0):
        _, unique_indices = np.unique(timestamps, return_index=True)
        timestamps = timestamps[unique_indices]
        data = data[unique_indices]
    
    duration = timestamps[-1] - timestamps[0]
    target_length = int(duration * target_fs)
    if target_length < 2: return np.array([]), np.array([])
    
    new_timestamps = np.linspace(timestamps[0], timestamps[-1], target_length)
    interp_func = interp1d(timestamps, data, kind='linear', bounds_error=False, fill_value='extrapolate')
    return interp_func(new_timestamps), new_timestamps

def find_major_troughs(data, sampling_rate, prominence_factor=0.25):
    prominence_threshold = max(np.std(data) * prominence_factor, 1.5)
    
    # 检测所有波谷
    all_troughs, _ = signal.find_peaks(-data, 
                                      distance=int(0.6 * sampling_rate), 
                                      prominence=prominence_threshold)
    
    return all_troughs

def process_segment_data(subject_folder, segment, subject_id, source_base_dir):
    bp_path = os.path.join(subject_folder, f'{segment}_bp.csv')
    if not os.path.exists(bp_path): return None
    try:
        # 读取BP数据
        df_bp = pd.read_csv(bp_path)
        if 'timestamp' not in df_bp.columns or 'bp' not in df_bp.columns:
            df_bp = pd.read_csv(bp_path, header=None)
            df_bp.columns = ['timestamp', 'bp'] if df_bp.shape[1] == 2 else ['timestamp', 'bp'] + [f'col_{i}' for i in range(2, df_bp.shape[1])]
        
        df_bp['timestamp'] = pd.to_numeric(df_bp['timestamp'], errors='coerce')
        df_bp['bp'] = pd.to_numeric(df_bp['bp'], errors='coerce')
        df_bp.dropna(subset=['timestamp', 'bp'], inplace=True)
        df_bp.drop_duplicates(subset=['timestamp'], inplace=True)
        df_bp = df_bp.sort_values(by='timestamp').reset_index(drop=True)
        if df_bp.empty: return None

        bp_data_raw = df_bp['bp'].values
        
        bp_data_filtered = butter_bandpass_filter_stable(bp_data_raw, config.BP_FILTER_LOW, config.BP_FILTER_HIGH, config.BP_SAMPLING_RATE)
        
        correction_offset = 0.0
        if segment == 'seg1':
            correction_offset = get_bp_correction_offset_from_files(subject_id, segment, source_base_dir)
        
        bp_corrected = bp_data_filtered + correction_offset
        
        bp_resampled, _ = resample_data(bp_corrected, df_bp['timestamp'].values, config.BP_SAMPLING_RATE, config.TARGET_SAMPLING_RATE)
        if len(bp_resampled) == 0: return None

        ppg_data_dict = {}
        for sensor in config.SENSORS_TO_USE:
            sensor_path = os.path.join(subject_folder, f'{segment}_{sensor}.csv')
            if not os.path.exists(sensor_path): continue
            
            df_sensor = pd.read_csv(sensor_path)
            if 'ir' not in df_sensor.columns or 'timestamp' not in df_sensor.columns:
                if df_sensor.shape[1] >= 2:
                    df_sensor = pd.read_csv(sensor_path, header=None)
                    df_sensor.columns = ['timestamp', 'ir'] + [f'col_{i}' for i in range(2, df_sensor.shape[1])]
                else:
                    continue
            
            df_sensor['timestamp'] = pd.to_numeric(df_sensor['timestamp'], errors='coerce')
            df_sensor['ir'] = pd.to_numeric(df_sensor['ir'], errors='coerce')
            df_sensor.dropna(subset=['timestamp', 'ir'], inplace=True)
            df_sensor.drop_duplicates(subset=['timestamp'], inplace=True)
            df_sensor = df_sensor.sort_values(by='timestamp').reset_index(drop=True)
            if df_sensor.empty: continue
            
            ppg_filtered = butter_bandpass_filter_stable(df_sensor['ir'].values, config.PPG_FILTER_LOW, config.PPG_FILTER_HIGH, config.SENSOR_SAMPLING_RATE)
            ppg_resampled, _ = resample_data(ppg_filtered, df_sensor['timestamp'].values, config.SENSOR_SAMPLING_RATE, config.TARGET_SAMPLING_RATE)
            if len(ppg_resampled) > 0:
                ppg_data_dict[sensor] = ppg_resampled
        
        if not ppg_data_dict: return None
        
        min_length = min(len(bp_resampled), min(len(data) for data in ppg_data_dict.values()))
        if min_length < config.WINDOW_SIZE: return None

        bp_aligned = bp_resampled[:min_length]
        ppg_aligned = np.zeros((len(config.SENSORS_TO_USE), min_length))
        sensor_mask = np.zeros(len(config.SENSORS_TO_USE), dtype=bool)
        for i, sensor in enumerate(config.SENSORS_TO_USE):
            if sensor in ppg_data_dict:
                ppg_aligned[i, :] = ppg_data_dict[sensor][:min_length]
                sensor_mask[i] = True
        return ppg_aligned, bp_aligned, sensor_mask
    except Exception as e:
        print(f"  - CRITICAL ERROR in process_segment_data for {subject_id}-{segment}: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_slices_from_troughs(ppg_data, bp_data, sensor_mask):
    if bp_data.shape[0] < config.WINDOW_SIZE: 
        return [], [], []
    
    major_troughs = find_major_troughs(bp_data, config.TARGET_SAMPLING_RATE)
    
    if len(major_troughs) < 2: 
        return [], [], []
    
    ppg_slices, bp_slices, mask_slices = [], [], []
    
    for i in range(len(major_troughs) - 1):
        start_idx = major_troughs[i]
        end_idx = start_idx + config.WINDOW_SIZE
        
        if end_idx > bp_data.shape[0]: 
            break
        
        next_trough_idx = major_troughs[i + 1]
        if next_trough_idx < end_idx:
            ppg_slice = ppg_data[:, start_idx:end_idx]
            bp_slice = bp_data[start_idx:end_idx]
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
        if not subjects: continue
        all_ppg, all_bp, all_masks = [], [], []
        print(f"\n--- Generating {split_name.upper()} SET ---")
        for subject_id in tqdm(subjects, desc=f"Processing {split_name} subjects"):
            subject_folder = os.path.join(config.RAW_DATA_DIR, subject_id)
            if not os.path.isdir(subject_folder): continue
            for segment in ['seg1', 'seg7']:
                result = process_segment_data(subject_folder, segment, subject_id, config.SHARED_DATA_DIR)
                if result is not None:
                    ppg, bp, mask = result
                    ppg_s, bp_s, mask_s = create_slices_from_troughs(ppg, bp, mask)
                    if ppg_s:
                        all_ppg.extend(ppg_s)
                        all_bp.extend(bp_s)
                        all_masks.extend(mask_s)
        if not all_ppg:
            print(f"No data generated for {split_name} set.")
            continue
        np.savez(os.path.join(config.PROCESSED_DATA_DIR, f"{split_name}_data.npz"),
                 ppg=np.array(all_ppg, dtype=np.float32),
                 bp=np.array(all_bp, dtype=np.float32),
                 sensor_mask=np.array(all_masks, dtype=bool))
        print(f"Generated and saved {len(all_ppg)} samples for {split_name} set.")

if __name__ == "__main__":
    main()