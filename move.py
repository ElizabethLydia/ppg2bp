import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   filename='bp_calibration.log',
                   filemode='w')

# Configuration - adjust these paths as needed
BASE_DIR = "/root/shared/PhysioNet2025"  # Correct path based on ls output
OUTPUT_DIR = "/root/shared/PhysioNet2025_Calibrated"  # Output directory

def load_omron_data():
    """
    Load Omron BP data from centralized omron.csv file.
    It assumes one row per subject, applying the same BP values to both seg1 and seg7.
    
    Returns:
        A dictionary mapping 'subjectID_segment' to (sbp, dbp) tuples.
        e.g., {'00042_seg1': (107, 77), '00042_seg7': (107, 77)}
    """
    possible_paths = [
        '/root/shared/omron.csv',
        '/root/ppg2bp/data/omron.csv',
        os.path.join(BASE_DIR, 'omron.csv'),
        os.path.join(os.path.dirname(BASE_DIR), 'omron.csv'),
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
                # Convert ID to integer first to remove ".0", then format to 5-digit string.
                subject_id = str(int(subject_id_raw)).strip().zfill(5)
                
                omron_data[f"{subject_id}_seg1"] = (sbp, dbp)
                omron_data[f"{subject_id}_seg7"] = (sbp, dbp)
                logging.debug(f"Loaded and stored Omron data for key '{subject_id}_seg1' and '{subject_id}_seg7'")
        
        return omron_data
        
    except Exception as e:
        logging.error(f"Failed to load or parse Omron data from '{omron_path}': {e}")
        return {}

def load_biopac_bp_values(biopac_folder, segment_num):
    """
    Load Biopac SBP and DBP from separate files in the same Biopac folder
    
    Args:
        biopac_folder: Path to Biopac folder
        segment_num: Segment number (1-11)
    
    Returns:
        (sbp_mean, dbp_mean) tuple or (None, None) if loading fails
    """
    try:
        # Look for systolic BP files
        systolic_files = glob.glob(os.path.join(biopac_folder, "*systolic*bp.csv"))
        if systolic_files:
            df_sbp = pd.read_csv(systolic_files[0], header=None, dtype=str)
            df_sbp.columns = ['timestamp', 'sbp']
            df_sbp['sbp'] = pd.to_numeric(df_sbp['sbp'], errors='coerce')
            df_sbp.dropna(inplace=True)
            # Filter out zero values
            df_sbp = df_sbp[df_sbp['sbp'] != 0]
            sbp_mean = df_sbp['sbp'].mean() if not df_sbp.empty else None
        else:
            logging.debug(f"Systolic BP file not found in: {biopac_folder}")
            sbp_mean = None
        
        # Look for diastolic BP files
        diastolic_files = glob.glob(os.path.join(biopac_folder, "*diastolic*bp.csv"))
        if diastolic_files:
            df_dbp = pd.read_csv(diastolic_files[0], header=None, dtype=str)
            df_dbp.columns = ['timestamp', 'dbp']
            df_dbp['dbp'] = pd.to_numeric(df_dbp['dbp'], errors='coerce')
            df_dbp.dropna(inplace=True)
            # Filter out zero values
            df_dbp = df_dbp[df_dbp['dbp'] != 0]
            dbp_mean = df_dbp['dbp'].mean() if not df_dbp.empty else None
        else:
            logging.debug(f"Diastolic BP file not found in: {biopac_folder}")
            dbp_mean = None
            
        if sbp_mean is not None and dbp_mean is not None:
            logging.debug(f"Biopac BP values for seg{segment_num}: SBP={sbp_mean:.1f}, DBP={dbp_mean:.1f} (zeros excluded)")
        
        return sbp_mean, dbp_mean
        
    except Exception as e:
        logging.error(f"Error loading Biopac BP values: {e}")
        return None, None

def calculate_bp_shift_amount(subject_id, task_1_folder):
    """
    Calculate BP shift amount based on seg1 Omron vs Biopac comparison
    
    Args:
        subject_id: Subject ID (e.g., '00017')
        task_1_folder: Task 1 folder path (e.g., '/path/to/00017/1')
    
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
    biopac_folder = os.path.join(task_1_folder, "Biopac")
    biopac_sbp, biopac_dbp = load_biopac_bp_values(biopac_folder, 1)
    
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

def apply_bp_calibration(bp_data, shift_amount):
    """
    Apply BP calibration using pre-calculated shift amount
    
    Args:
        bp_data: BP waveform data
        shift_amount: Pre-calculated shift amount, or None for no shift
    
    Returns:
        Calibrated BP data
    """
    if shift_amount is not None:
        calibrated_bp = bp_data + shift_amount
        return calibrated_bp
    else:
        return bp_data

def save_calibrated_bp(bp_data, original_timestamps, output_path):
    """
    Save calibrated BP data in original CSV format
    
    Args:
        bp_data: Calibrated BP waveform data
        original_timestamps: Original timestamps from raw data
        output_path: Output file path
    """
    # Create output directory if not exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV with exact same format as original
    with open(output_path, 'w') as f:
        for timestamp, bp_value in zip(original_timestamps, bp_data):
            # Format to match original: preserve timestamp precision, round BP to 4 decimal places
            f.write(f"{timestamp:.7f},{bp_value:.4f}\n")
    
    logging.debug(f"Saved calibrated BP data to: {output_path}")

def process_subject_task(subject_id, task_id, task_folder, shift_amount, output_base_dir):
    """
    Process BP data for a specific subject and task
    
    Args:
        subject_id: Subject ID (e.g., '00017')
        task_id: Task ID ('1' to '11')
        task_folder: Path to task folder
        shift_amount: Pre-calculated shift amount
        output_base_dir: Base output directory
    """
    segment_name = f'seg{task_id}'
    
    # Find BP file in Biopac folder - handle both bp.csv and bp-{task_id}.csv formats
    biopac_folder = os.path.join(task_folder, "Biopac")
    
    # Try different BP file naming patterns
    bp_file_patterns = [
        os.path.join(biopac_folder, "bp.csv"),           # Standard format
        os.path.join(biopac_folder, f"bp-{task_id}.csv") # Alternative format
    ]
    
    bp_file = None
    for pattern in bp_file_patterns:
        if os.path.exists(pattern):
            bp_file = pattern
            break
    
    if bp_file is None:
        logging.warning(f"BP file not found in {biopac_folder} (tried: bp.csv, bp-{task_id}.csv)")
        return
    
    try:
        # Read original BP data
        df_bp = pd.read_csv(bp_file, header=None, dtype=str)
        df_bp.columns = ['timestamp', 'bp']
        
        df_bp['timestamp'] = pd.to_numeric(df_bp['timestamp'], errors='coerce')
        df_bp['bp'] = pd.to_numeric(df_bp['bp'], errors='coerce')
        df_bp.dropna(inplace=True)
        
        if df_bp.empty:
            logging.warning(f"No valid BP data in {bp_file}")
            return
            
        df_bp = df_bp.sort_values(by='timestamp').reset_index(drop=True)
        
        # Apply calibration
        original_timestamps = df_bp['timestamp'].values
        original_bp = df_bp['bp'].values
        calibrated_bp = apply_bp_calibration(original_bp, shift_amount)
        
        # Create output path: output_dir/subject_id/task_id/Biopac/bp.csv
        output_path = os.path.join(output_base_dir, subject_id, task_id, "Biopac", "bp.csv")
        
        # Save calibrated data
        save_calibrated_bp(calibrated_bp, original_timestamps, output_path)
        
        shift_info = f"{shift_amount:+.1f} mmHg" if shift_amount is not None else "No shift"
        print(f"  {segment_name}: {shift_info} ({os.path.basename(bp_file)})")
        
    except Exception as e:
        logging.error(f"Error processing {subject_id}/{task_id}: {e}")
        raise

def main():
    """
    Main function to process all subjects and tasks
    """
    print("BP Calibration Script")
    print(f"Source directory: {BASE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Find all date folders
    date_folders = glob.glob(os.path.join(BASE_DIR, "202*"))
    
    if not date_folders:
        print(f"No date folders found in {BASE_DIR}")
        return
    
    all_subjects = set()
    subject_paths = {}
    
    # Collect all subjects from all date folders
    for date_folder in date_folders:
        subjects_in_date = glob.glob(os.path.join(date_folder, "*"))
        for subject_path in subjects_in_date:
            if os.path.isdir(subject_path):
                subject_id = os.path.basename(subject_path)
                all_subjects.add(subject_id)
                subject_paths[subject_id] = subject_path
    
    all_subjects = sorted(list(all_subjects))
    print(f"Found {len(all_subjects)} subjects: {all_subjects}")
    
    # Track processing results
    successful_subjects = []
    failed_subjects = []
    partial_subjects = []  # Subjects with some tasks processed
    
    # Process each subject
    for subject_id in tqdm(all_subjects, desc="Processing subjects"):
        subject_folder = subject_paths[subject_id]
        print(f"\nProcessing {subject_id}:")
        logging.info(f"Starting processing for subject {subject_id}")
        
        # Find task folders (1 to 11)
        task_folders = {}
        for task_id in range(1, 12):  # 1 to 11
            task_id_str = str(task_id)
            task_path = os.path.join(subject_folder, task_id_str)
            if os.path.isdir(task_path):
                task_folders[task_id_str] = task_path
        
        if not task_folders:
            print(f"  No task folders found for {subject_id}")
            logging.error(f"FAILED: No task folders found for subject {subject_id}")
            failed_subjects.append(subject_id)
            continue
        
        # Calculate shift amount based on task 1 (seg1)
        shift_amount = None
        if '1' in task_folders:
            shift_amount = calculate_bp_shift_amount(subject_id, task_folders['1'])
            if shift_amount is not None:
                logging.info(f"Calculated shift amount for {subject_id}: {shift_amount:+.1f} mmHg")
            else:
                logging.warning(f"Could not calculate shift amount for {subject_id}")
        
        # Process each available task
        tasks_processed = 0
        tasks_total = len(task_folders)
        
        for task_id_str, task_folder in task_folders.items():
            try:
                process_subject_task(subject_id, task_id_str, task_folder, shift_amount, OUTPUT_DIR)
                tasks_processed += 1
                logging.info(f"Successfully processed {subject_id}/task{task_id_str}")
            except Exception as e:
                logging.error(f"Failed to process {subject_id}/task{task_id_str}: {e}")
        
        # Categorize processing result
        if tasks_processed == 0:
            failed_subjects.append(subject_id)
            logging.error(f"FAILED: No tasks processed for subject {subject_id}")
        elif tasks_processed == tasks_total:
            successful_subjects.append(subject_id)
            logging.info(f"SUCCESS: All {tasks_processed} tasks processed for subject {subject_id}")
        else:
            partial_subjects.append(subject_id)
            logging.warning(f"PARTIAL: {tasks_processed}/{tasks_total} tasks processed for subject {subject_id}")
    
    # Print final summary
    print(f"\nCalibration complete! Results saved to: {OUTPUT_DIR}")
    print(f"\n=== PROCESSING SUMMARY ===")
    print(f"Total subjects: {len(all_subjects)}")
    print(f"Successful: {len(successful_subjects)}")
    print(f"Partial: {len(partial_subjects)}")
    print(f"Failed: {len(failed_subjects)}")
    
    if successful_subjects:
        print(f"\nSuccessful subjects ({len(successful_subjects)}): {successful_subjects}")
        logging.info(f"SUMMARY - Successful subjects: {successful_subjects}")
    
    if partial_subjects:
        print(f"\nPartial subjects ({len(partial_subjects)}): {partial_subjects}")
        logging.warning(f"SUMMARY - Partial subjects: {partial_subjects}")
    
    if failed_subjects:
        print(f"\nFailed subjects ({len(failed_subjects)}): {failed_subjects}")
        logging.error(f"SUMMARY - Failed subjects: {failed_subjects}")
    
    print("\nOutput structure:")
    print("PhysioNet2025_Calibrated/")
    for subject_id in successful_subjects[:3]:  # Show first 3 successful as example
        output_subject_dir = os.path.join(OUTPUT_DIR, subject_id)
        if os.path.exists(output_subject_dir):
            print(f"  {subject_id}/")
            for task_id in range(1, 12):  # 1 to 11
                task_id_str = str(task_id)
                task_output_dir = os.path.join(output_subject_dir, task_id_str, "Biopac")
                if os.path.exists(task_output_dir):
                    print(f"    {task_id_str}/")
                    print(f"      Biopac/")
                    print(f"        bp.csv")
    if len(successful_subjects) > 3:
        print("  ...")

if __name__ == "__main__":
    main()