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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def decompress_archives(source_dir):
    print(f"Scanning for archives in: {source_dir}")
    archives = glob.glob(os.path.join(source_dir, "*.tar")) + glob.glob(os.path.join(source_dir, "*.tar.gz"))
    
    if not archives:
        print("No archives found to decompress.")
        return

    for archive_path in archives:
        archive_name = os.path.basename(archive_path)
        extracted_folder_name = archive_name.replace(".tar.gz", "").replace(".tar", "")
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

def organize_raw_data(source_base_dir, dest_dir, subjects_to_process):
    print(f"Organizing raw data from: {source_base_dir}")
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
        found_files_for_subject = False

        for task_id, seg_name in task_mapping.items():
            task_folder = os.path.join(subject_path, task_id)
            if not os.path.isdir(task_folder):
                continue

            biopac_folder = os.path.join(task_folder, "Biopac")
            bp_src_path = os.path.join(biopac_folder, "bp.csv")
            if os.path.exists(bp_src_path):
                bp_dest_path = os.path.join(subject_dest_folder, f"{seg_name}_bp.csv")
                shutil.copy2(bp_src_path, bp_dest_path)
                found_files_for_subject = True

            hub_folder = os.path.join(task_folder, "HUB")
            if os.path.isdir(hub_folder):
                sensor_files = glob.glob(os.path.join(hub_folder, "*.csv"))
                for sensor_src_path in sensor_files:
                    original_filename = os.path.basename(sensor_src_path)
                    sensor_dest_path = os.path.join(subject_dest_folder, f"{seg_name}_{original_filename}")
                    shutil.copy2(sensor_src_path, sensor_dest_path)
                    found_files_for_subject = True

        if found_files_for_subject:
             print(f"Organized data for {subject_id}")

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    if len(data) < order * 3:  # 数据太短，不滤波
        print(f"Warning: Data too short for filtering ({len(data)} points), skipping filter")
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
        print(f"Warning: Invalid frequency range (low={low}, high={high}), skipping filter")
        return data
        
    try:
        b, a = signal.butter(order, [low, high], btype='band')
        filtered_data = signal.filtfilt(b, a, data)
        
        # 检查滤波结果
        if np.all(filtered_data == 0) or np.all(np.isnan(filtered_data)):
            print("Warning: Filtering resulted in all zeros/NaNs, returning original data")
            return data
            
        print(f"Filtering: {len(data)} points, range before=({np.min(data):.4f}, {np.max(data):.4f}), after=({np.min(filtered_data):.4f}, {np.max(filtered_data):.4f})")
        return filtered_data
        
    except Exception as e:
        print(f"Warning: Filtering failed ({e}), returning original data")
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
        print(f"Resampling error: {e}")
        return np.array([]), np.array([])

def normalize_signal(data):
    if len(data) == 0:
        return data
    
    # 检查数据是否全为相同值
    if np.all(data == data[0]):
        print(f"Warning: Signal is constant (value={data[0]}), skipping normalization")
        return data - np.mean(data)  # 只去均值
    
    mean_val = np.mean(data)
    std_val = np.std(data)
    
    if std_val < 1e-8:
        print(f"Warning: Very small std ({std_val}), using robust normalization")
        # 使用更鲁棒的归一化方法
        median_val = np.median(data)
        mad = np.median(np.abs(data - median_val))
        if mad > 1e-8:
            return (data - median_val) / mad
        else:
            return data - mean_val
    
    normalized = (data - mean_val) / std_val
    print(f"Normalization: mean={mean_val:.4f}, std={std_val:.4f}, range=({np.min(normalized):.4f}, {np.max(normalized):.4f})")
    return normalized

def process_segment_data(subject_folder, segment):
    bp_path = os.path.join(subject_folder, f'{segment}_bp.csv')
    if not os.path.exists(bp_path):
        return None
    
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
            
        # 【修改】不再调用 bp_normalized = normalize_signal(bp_resampled)
        
        ppg_data_dict = {}
        available_sensors = []
        
        for sensor in config.SENSORS_TO_USE:
            sensor_path = os.path.join(subject_folder, f'{segment}_{sensor}.csv')
            if not os.path.exists(sensor_path):
                print(f"Warning: {sensor_path} not found. Skipping this sensor.")
                continue
            
            df_sensor = pd.read_csv(sensor_path)
            if 'ir' not in df_sensor.columns or 'timestamp' not in df_sensor.columns:
                print(f"Warning: Required columns missing in {sensor_path}")
                continue
                
            df_sensor['timestamp'] = pd.to_numeric(df_sensor['timestamp'], errors='coerce')
            df_sensor['ir'] = pd.to_numeric(df_sensor['ir'], errors='coerce')
            df_sensor.dropna(inplace=True)
            
            if df_sensor.empty:
                print(f"Warning: No valid data in {sensor_path}")
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
                    print(f"Warning: Resampling failed for {sensor}")
                    continue
                    
                ppg_data_dict[sensor] = ppg_resampled 
                available_sensors.append(sensor)
                
            except Exception as e:
                print(f"Warning: Error processing {sensor}: {e}")
                continue
        
        if len(available_sensors) == 0:
            print(f"Warning: No valid sensors for {segment}")
            return None
        
        print(f"Available sensors for {segment}: {available_sensors} ({len(available_sensors)}/{len(config.SENSORS_TO_USE)})")
        
        min_length = min(len(bp_resampled), min(len(data) for data in ppg_data_dict.values()))
        
        if min_length < config.WINDOW_SIZE:
            print(f"Warning: Insufficient data length ({min_length}) for {segment}")
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
        print(f"Error processing {segment}: {e}")
        return None

def create_slices(ppg_data, bp_data, sensor_mask):
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
        
        print(f"Generating {split_name.upper()} SET")
        for subject_id in tqdm(subjects, desc=f"Processing {split_name} subjects"):
            subject_folder = os.path.join(config.RAW_DATA_DIR, subject_id)
            if not os.path.isdir(subject_folder):
                print(f"Warning: Directory for subject {subject_id} not found.")
                continue

            for segment in ['seg1', 'seg7']:
                print(f"Processing {subject_id} - {segment}")
                result = process_segment_data(subject_folder, segment)
                
                if result is not None:
                    ppg_data, bp_data, sensor_mask = result
                    ppg_slices, bp_slices, mask_slices = create_slices(ppg_data, bp_data, sensor_mask)
                    
                    if ppg_slices:
                        all_ppg_for_split.extend(ppg_slices)
                        all_bp_for_split.extend(bp_slices)
                        all_masks_for_split.extend(mask_slices)
                        print(f"Generated {len(ppg_slices)} slices from {segment}")
                    else:
                        print(f"No valid slices from {segment}")
                else:
                    print(f"No data processed for {segment}")

        if not all_ppg_for_split:
            print(f"No data generated for {split_name} set.")
            continue

        all_ppg_np = np.array(all_ppg_for_split, dtype=np.float32)
        all_bp_np = np.array(all_bp_for_split, dtype=np.float32)
        all_masks_np = np.array(all_masks_for_split, dtype=bool)
        
        print(f"Generated {len(all_ppg_np)} samples for {split_name} set")
        print(f"PPG shape: {all_ppg_np.shape}, BP shape: {all_bp_np.shape}, Mask shape: {all_masks_np.shape}")
        
        # 统计传感器可用率
        sensor_availability = np.mean(all_masks_np, axis=0)
        for i, sensor in enumerate(config.SENSORS_TO_USE):
            print(f"  {sensor} availability: {sensor_availability[i]*100:.1f}%")
        
        save_path_npz = os.path.join(config.PROCESSED_DATA_DIR, f"{split_name}_data.npz")
        np.savez(save_path_npz, ppg=all_ppg_np, bp=all_bp_np, sensor_mask=all_masks_np)
        print(f"Saved to {save_path_npz}")

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