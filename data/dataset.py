import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import numpy as np
from torch.utils.data import Dataset

class PPGDataset(Dataset):
    def __init__(self, npz_path):
        try:
            data = np.load(npz_path)
            self.ppg_data = data['ppg']
            self.bp_data = data['bp']
            
            if 'sensor_mask' in data:
                self.sensor_mask = data['sensor_mask']
                print(f"Loaded dataset with sensor masks. Shape: {self.sensor_mask.shape}")
            else:
                print("No sensor masks found, assuming all sensors are available")
                self.sensor_mask = np.ones((len(self.ppg_data), self.ppg_data.shape[1]), dtype=bool)
            
            if len(self.ppg_data) == 0 or len(self.bp_data) == 0:
                raise ValueError("Empty dataset loaded")
                
            if len(self.ppg_data) != len(self.bp_data):
                min_len = min(len(self.ppg_data), len(self.bp_data))
                self.ppg_data = self.ppg_data[:min_len]
                self.bp_data = self.bp_data[:min_len]
                self.sensor_mask = self.sensor_mask[:min_len]
                print(f"Warning: Mismatched lengths, truncated to {min_len}")
                
        except FileNotFoundError:
            print(f"Error: Data file not found at {npz_path}")
            print("Please run data/generate_dataset.py first.")
            self.ppg_data = np.array([])
            self.bp_data = np.array([])
            self.sensor_mask = np.array([])
        except Exception as e:
            print(f"Error loading dataset: {e}")
            self.ppg_data = np.array([])
            self.bp_data = np.array([])
            self.sensor_mask = np.array([])

        if len(self.ppg_data) > 0:
            self.ppg_data = torch.from_numpy(self.ppg_data).float()
            self.bp_data = torch.from_numpy(self.bp_data).float()
            self.sensor_mask = torch.from_numpy(self.sensor_mask).bool()
            
            if self.bp_data.ndim == 2:
                self.bp_data = self.bp_data.unsqueeze(1)
            elif self.bp_data.ndim == 1:
                self.bp_data = self.bp_data.unsqueeze(0).unsqueeze(0)

    def __len__(self):
        return len(self.ppg_data) if len(self.ppg_data) > 0 else 0

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("Index out of dataset range")
            
        ppg_sample = self.ppg_data[idx]
        bp_sample = self.bp_data[idx]
        mask_sample = self.sensor_mask[idx]
        
        if torch.isnan(ppg_sample).any() or torch.isnan(bp_sample).any():
            print(f"Warning: NaN values found in sample {idx}")
            ppg_sample = torch.nan_to_num(ppg_sample)
            bp_sample = torch.nan_to_num(bp_sample)
        
        return ppg_sample, bp_sample, mask_sample