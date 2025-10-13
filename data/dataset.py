import os
import numpy as np
import torch
from torch.utils.data import Dataset

class PPGDataset(Dataset):
    def __init__(self, npz_path: str, mmap: bool = True, make_contiguous: bool = True):
        self.ppg_data = torch.empty(0)
        self.bp_data = torch.empty(0)
        self.sensor_mask = torch.empty(0, dtype=torch.bool)

        if not os.path.exists(npz_path):
            print(f"[PPGDataset] Error: file not found -> {npz_path}")
            return

        try:
            # 关键：用上下文管理器，读完立刻关闭文件句柄，避免多进程卡死
            with np.load(npz_path, mmap_mode=('r' if mmap else None)) as store:
                if 'ppg' not in store.files or 'bp' not in store.files:
                    raise ValueError(f"NPZ missing required arrays 'ppg' or 'bp'. Found: {store.files}")

                ppg = store['ppg']       # 期望形状 (N, C, L)
                bp = store['bp']         # 期望形状 (N, L) 或 (N, 1, L)

                if 'sensor_mask' in store.files:
                    sensor_mask = store['sensor_mask']  # 期望形状 (N, C)
                else:
                    sensor_mask = None

            # ----------- 基本一致性处理 -----------
            N = min(len(ppg), len(bp))
            if N == 0:
                print("[PPGDataset] Error: empty arrays in NPZ.")
                return

            ppg = ppg[:N]
            bp = bp[:N]
            # ppg: (N, C, L)
            if ppg.ndim != 3:
                raise ValueError(f"ppg array must be 3D (N,C,L), got shape {ppg.shape}")

            N, C, L = ppg.shape

            # If single-sensor mode enabled in config, select only that channel
            try:
                import config as _config
                if getattr(_config, 'SINGLE_SENSOR_MODE', False):
                    si = int(getattr(_config, 'SINGLE_SENSOR_INDEX', 0))
                    if si < 0 or si >= C:
                        raise IndexError(f"SINGLE_SENSOR_INDEX={si} out of range for C={C}")
                    # slice to (N,1,L)
                    ppg = ppg[:, si:si+1, :]
                    C = 1
                    if sensor_mask is not None:
                        # keep mask aligned
                        sensor_mask = sensor_mask[:, si:si+1]
            except Exception:
                # Fail-safe: if config import or attribute missing, continue normally
                pass

            # bp: 统一到 (N, 1, L)
            if bp.ndim == 2:         # (N, L)
                if bp.shape[1] != L:
                    raise ValueError(f"bp length mismatch: bp.shape[1]={bp.shape[1]} vs L={L}")
                bp = bp[:, None, :]  # -> (N, 1, L)
            elif bp.ndim == 3:       # (N, 1, L) or (N, ?, L)
                if bp.shape[-1] != L:
                    raise ValueError(f"bp length mismatch: bp.shape[-1]={bp.shape[-1]} vs L={L}")
                if bp.shape[1] != 1:
                    # 如果是 (N, K, L) 意味着有多个通道，保留第一个
                    bp = bp[:, :1, :]
            else:
                raise ValueError(f"bp array must be 2D or 3D, got shape {bp.shape}")

            # sensor_mask: 统一到 (N, C)
            if sensor_mask is None:
                sensor_mask = np.ones((N, C), dtype=bool)
            else:
                if sensor_mask.ndim != 2 or sensor_mask.shape[0] != N:
                    # 对齐 N
                    sensor_mask = sensor_mask[:N]
                if sensor_mask.shape[1] != C:
                    # 如果 mask 通道数不匹配，尝试广播或截断
                    new_mask = np.ones((N, C), dtype=bool)
                    min_c = min(C, sensor_mask.shape[1])
                    new_mask[:, :min_c] = sensor_mask[:, :min_c].astype(bool)
                    sensor_mask = new_mask

            # ----------- 转成 torch.Tensor -----------
            # 可选：先复制为常规内存数组，避免 memmap 在多进程下的开销
            if make_contiguous:
                ppg = np.ascontiguousarray(ppg)
                bp = np.ascontiguousarray(bp)
                sensor_mask = np.ascontiguousarray(sensor_mask)

            self.ppg_data = torch.from_numpy(ppg).float()                 # (N, C, L)
            self.bp_data = torch.from_numpy(bp).float()                   # (N, 1, L)
            self.sensor_mask = torch.from_numpy(sensor_mask).bool()       # (N, C)

        except Exception as e:
            print(f"[PPGDataset] Error loading dataset: {e}")
            # 留空，__len__ 会返回 0

    def __len__(self):
        return int(self.ppg_data.shape[0]) if self.ppg_data.ndim == 3 else 0

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        ppg = self.ppg_data[idx]            # (C, L)
        bp = self.bp_data[idx]              # (1, L)
        mask = self.sensor_mask[idx]        # (C,)

        # 简单的 NaN 保护（如需）
        if torch.isnan(ppg).any():
            ppg = torch.nan_to_num(ppg)
        if torch.isnan(bp).any():
            bp = torch.nan_to_num(bp)

        return ppg, bp, mask