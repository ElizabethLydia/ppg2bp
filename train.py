import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os
import swanlab
import numpy as np
import config
from data.dataset import PPGDataset
from model.model import UNet1D
# --- MODIFICATION: Import the new attention-based loss function ---
from loss.trend_loss import CombinedBPAttentionLoss

# --- NEW: lightweight quality weighting using PPG HR consistency (TD vs FD) ---
import math
try:
    from scipy.signal import find_peaks
except Exception:
    find_peaks = None

def _hr_td_numpy(x: np.ndarray, fs: int) -> float:
    """Estimate HR from peaks (time-domain); return bpm or nan."""
    if x.size < max(3, int(0.6*fs)):
        return float('nan')
    # z-score normalize for stability
    x = (x - x.mean()) / (x.std() + 1e-8)
    if find_peaks is None:
        return float('nan')
    distance = max(1, int(0.4 * fs))  # >=150 bpm upper bound spacing
    peaks, _ = find_peaks(x, distance=distance)
    if len(peaks) < 3:
        return float('nan')
    rr = np.diff(peaks) / float(fs)
    rr = rr[(rr > 0.25) & (rr < 1.2)]  # 50–240 bpm range
    if rr.size == 0:
        return float('nan')
    return float(60.0 / np.median(rr))

def _hr_fd_numpy(x: np.ndarray, fs: int, hr_range_hz=(0.7, 2.5)) -> float:
    """Estimate HR from dominant FFT peak within HR band; return bpm or nan."""
    L = x.size
    if L < 4:
        return float('nan')
    x = x - x.mean()
    spec = np.fft.rfft(x)
    mag = np.abs(spec)
    freqs = np.fft.rfftfreq(L, d=1.0/fs)
    mask = (freqs >= hr_range_hz[0]) & (freqs <= hr_range_hz[1])
    if mask.sum() < 2:
        return float('nan')
    k = np.argmax(mag[mask])
    f0 = freqs[mask][k]
    return float(f0 * 60.0)

def _weights_from_ppg_consistency(ppg: torch.Tensor,
                                  sensor_mask: torch.Tensor | None,
                                  fs: int,
                                  strategy: str = 'pf',
                                  topk: int = 2,
                                  mode: str = 'weight',
                                  threshold: float = 0.4,
                                  min_weight: float = 0.2,
                                  alpha_bpm: float = 5.0) -> torch.Tensor:
    """
    Compute per-sample weights from PPG HR consistency.
    strategy:
      - 'pf': |HR_time - HR_freq| 作为差值（推荐）
      - 'pt': 预留：|HR_time - HR_true|（需要外部真HR，当前未启用）
    """
    B, C, L = ppg.shape
    x = ppg.detach().cpu().numpy()
    sm = sensor_mask.detach().cpu().numpy() if sensor_mask is not None else np.ones((B, C), dtype=bool)
    fs_val = int(fs)
    hr_band = getattr(config, 'SNR_HR_RANGE', (0.7, 2.5))

    q = np.zeros((B, C), dtype=np.float32)
    for b in range(B):
        for c in range(C):
            if not sm[b, c]:
                q[b, c] = 0.0
                continue
            sig = x[b, c, :]
            hr_td = _hr_td_numpy(sig, fs_val)
            hr_fd = _hr_fd_numpy(sig, fs_val, hr_band)
            if strategy == 'pf':
                if np.isfinite(hr_td) and np.isfinite(hr_fd):
                    delta = abs(hr_td - hr_fd)
                else:
                    delta = np.inf
            else:
                # 'pt' reserved: fallback to pf if true HR not integrated
                if np.isfinite(hr_td) and np.isfinite(hr_fd):
                    delta = abs(hr_td - hr_fd)
                else:
                    delta = np.inf
            if not np.isfinite(delta):
                q[b, c] = 0.0
            else:
                q[b, c] = float(np.exp(-delta / max(alpha_bpm, 1e-6)))
                # val = 1.0 - (delta / max(alpha_bpm, 1e-6)) ** 2
                # q[b, c] = float(val) if val > 0.0 else 0.0

    # top-k channel averaging
    k_eff = max(1, min(int(topk), C))
    # sort descending
    topk_vals = np.sort(q, axis=1)[:, -k_eff:]
    w = topk_vals.mean(axis=1)

    if mode == 'filter':
        w = np.where(w >= threshold, w, 0.0)
    else:
        w = np.clip(w, a_min=min_weight, a_max=None)

    # ensure tensor on same device as ppg
    return torch.from_numpy(w).to(device=ppg.device, dtype=ppg.dtype)

# Reproducibility utilities
import random

def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    
    # --- MODIFICATION: Update running totals for new loss components ---
    running_total_loss = 0.0
    running_sbp_dbp_loss = 0.0
    running_trend_loss = 0.0
    running_notch_loss = 0.0
    # --- NEW: SQI stats ---
    total_samples = 0
    kept_samples = 0
    weight_sum = 0.0
    
    for batch_data in tqdm(dataloader, desc="Training"):
        if len(batch_data) == 3:
            ppg, bp, sensor_mask = batch_data
            ppg, bp, sensor_mask = ppg.to(device), bp.to(device), sensor_mask.to(device)
        else: 
            ppg, bp = batch_data
            ppg, bp = ppg.to(device), bp.to(device)
            sensor_mask = None
        
        optimizer.zero_grad()
        
        predictions = model(ppg, sensor_mask)

        # --- NEW: compute sample weights via PPG HR consistency (TD vs FD) ---
        weights = None
        if getattr(config, 'USE_SQI', False):
            try:
                weights = _weights_from_ppg_consistency(
                    ppg=ppg,
                    sensor_mask=sensor_mask,
                    fs=getattr(config, 'TARGET_SAMPLING_RATE', 100),
                    strategy=getattr(config, 'SQI_STRATEGY', 'pf'),
                    topk=getattr(config, 'CHANNEL_PASS_K', 2),
                    mode=getattr(config, 'SQI_MODE', 'weight'),
                    threshold=getattr(config, 'SQI_THRESHOLD', 0.4),
                    min_weight=getattr(config, 'SAMPLE_WEIGHT_MIN', 0.2),
                    alpha_bpm=getattr(config, 'ALPHA_BPM', 8.0),
                )
            except Exception as _:
                weights = None
        
        # Update SQI stats and optionally skip if all filtered
        batch_size = ppg.size(0)
        total_samples += batch_size
        if weights is not None:
            if getattr(config, 'SQI_MODE', 'weight') == 'filter':
                kept = int((weights > 0).sum().item())
                kept_samples += kept
                weight_sum += float(weights.sum().item())
                if kept == 0:
                    # All filtered, skip backprop
                    optimizer.zero_grad(set_to_none=True)
                    continue
            else:
                kept_samples += batch_size
                weight_sum += float(weights.sum().item())
        else:
            kept_samples += batch_size

        # --- MODIFICATION: Unpack new loss and details dictionary ---
        loss, loss_details = criterion(predictions, bp, sample_weight=weights)
        
        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: NaN or Inf loss detected, skipping batch")
            continue
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
    # --- MODIFICATION: Accumulate new loss components ---
        batch_size = ppg.size(0)
        running_total_loss += loss.item() * batch_size
        running_sbp_dbp_loss += loss_details['sbp_dbp_loss'] * batch_size
        running_trend_loss += loss_details['trend_loss'] * batch_size
        running_notch_loss += loss_details['notch_loss'] * batch_size
        
    dataset_size = len(dataloader.dataset)
    
    # --- MODIFICATION: Return dictionary with new metrics ---
    mean_weight = (weight_sum / kept_samples) if kept_samples > 0 else 0.0
    filtered_ratio = 1.0 - (kept_samples / total_samples) if total_samples > 0 else 0.0
    return {
        'total_loss': running_total_loss / dataset_size,
        'sbp_dbp_loss': running_sbp_dbp_loss / dataset_size,
        'trend_loss': running_trend_loss / dataset_size,
        'notch_loss': running_notch_loss / dataset_size,
        'mean_weight': mean_weight,
        'filtered_ratio': filtered_ratio
    }

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()

    # --- MODIFICATION: Update running totals for new loss components ---
    running_total_loss = 0.0
    running_sbp_dbp_loss = 0.0
    running_trend_loss = 0.0
    running_notch_loss = 0.0
    # --- NEW: SQI stats ---
    total_samples = 0
    kept_samples = 0
    weight_sum = 0.0
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Validating"):
            if len(batch_data) == 3:
                ppg, bp, sensor_mask = batch_data
                ppg, bp, sensor_mask = ppg.to(device), bp.to(device), sensor_mask.to(device)
            else:
                ppg, bp = batch_data
                ppg, bp = ppg.to(device), bp.to(device)
                sensor_mask = None
                
            predictions = model(ppg, sensor_mask)

            # --- NEW: compute validation weights (same rule) ---
            weights = None
            if getattr(config, 'USE_SQI', False):
                try:
                    weights = _weights_from_ppg_consistency(
                        ppg=ppg,
                        sensor_mask=sensor_mask,
                        fs=getattr(config, 'TARGET_SAMPLING_RATE', 100),
                        strategy=getattr(config, 'SQI_STRATEGY', 'pf'),
                        topk=getattr(config, 'CHANNEL_PASS_K', 2),
                        mode=getattr(config, 'SQI_MODE', 'weight'),
                        threshold=getattr(config, 'SQI_THRESHOLD', 0.4),
                        min_weight=getattr(config, 'SAMPLE_WEIGHT_MIN', 0.2),
                        alpha_bpm=getattr(config, 'ALPHA_BPM', 8.0),
                    )
                except Exception as _:
                    weights = None
            
            # Update SQI stats and optionally skip if all filtered
            batch_size = ppg.size(0)
            total_samples += batch_size
            if weights is not None:
                if getattr(config, 'SQI_MODE', 'weight') == 'filter':
                    kept = int((weights > 0).sum().item())
                    kept_samples += kept
                    weight_sum += float(weights.sum().item())
                    if kept == 0:
                        continue
                else:
                    kept_samples += batch_size
                    weight_sum += float(weights.sum().item())
            else:
                kept_samples += batch_size

            # --- MODIFICATION: Unpack new loss and details dictionary ---
            loss, loss_details = criterion(predictions, bp, sample_weight=weights)
            
            if not torch.isnan(loss) and not torch.isinf(loss):
                # --- MODIFICATION: Accumulate new loss components ---
                batch_size = ppg.size(0)
                running_total_loss += loss.item() * batch_size
                running_sbp_dbp_loss += loss_details['sbp_dbp_loss'] * batch_size
                running_trend_loss += loss_details['trend_loss'] * batch_size
                running_notch_loss += loss_details['notch_loss'] * batch_size

    dataset_size = len(dataloader.dataset)
    
    # --- MODIFICATION: Return dictionary with new metrics ---
    mean_weight = (weight_sum / kept_samples) if kept_samples > 0 else 0.0
    filtered_ratio = 1.0 - (kept_samples / total_samples) if total_samples > 0 else 0.0
    return {
        'total_loss': running_total_loss / dataset_size,
        'sbp_dbp_loss': running_sbp_dbp_loss / dataset_size,
        'trend_loss': running_trend_loss / dataset_size,
        'notch_loss': running_notch_loss / dataset_size,
        'mean_weight': mean_weight,
        'filtered_ratio': filtered_ratio
    }

def main():
    # Phase 0: enforce reproducibility
    set_seed(getattr(config, "SEED", 42))
    # Create a unique run directory: runs/<name>_<timestamp>/
    from datetime import datetime
    run_name = f"unet1d_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(getattr(config, 'RUNS_DIR', './runs'), run_name)
    run_results = os.path.join(run_dir, 'results')
    run_models = os.path.join(run_dir, 'saved_models')
    run_logs = os.path.join(run_dir, 'logs')
    for d in (run_dir, run_results, run_models, run_logs):
        os.makedirs(d, exist_ok=True)
    swanlab.init(
    project="ppg2bp-train",
    name=run_name,
    config={
        "epochs": config.NUM_EPOCHS,
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "dropout": 0.2,
        "loss_weights": {
            "sbp_dbp": 0.4,
            "trend": 0.4,
            "notch": 0.2
            }
        }
    )
    print(f"Loading datasets... (run_dir={run_dir})")
    train_dataset = PPGDataset(os.path.join(config.PROCESSED_DATA_DIR, "train_seg1_data.npz"))
    val_dataset = PPGDataset(os.path.join(config.PROCESSED_DATA_DIR, "validation_seg1_data.npz"))
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Error: Train or validation dataset is empty.")
        return

    def _worker_init_fn(worker_id):
        seed = (getattr(config, "SEED", 42) + worker_id) % (2**32 - 1)
        np.random.seed(seed)
        random.seed(seed)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, 
                             num_workers=4, pin_memory=True, drop_last=True, worker_init_fn=_worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, 
                           num_workers=4, pin_memory=True, drop_last=False, worker_init_fn=_worker_init_fn)
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    print(f"Initializing model on device: {config.DEVICE}")
    model = UNet1D(in_channels=config.IN_CHANNELS, output_points=config.OUTPUT_POINTS, dropout_rate=0.2).to(config.DEVICE)
    
    # --- MODIFICATION: Initialize the new CombinedBPAttentionLoss ---
    criterion = CombinedBPAttentionLoss(
        sbp_dbp_weight=0.4, 
        trend_weight=0.4, 
        notch_weight=0.2,
        sbp_dbp_mode=getattr(config, 'SBP_DBP_MODE', 'mse'),
        huber_delta=getattr(config, 'HUBER_DELTA', 1.0),
        mix_alpha=getattr(config, 'MIX_ALPHA', 0.5)
    )
    print("Using CombinedBPAttentionLoss with weights - SBP/DBP: 40%, Trend: 40%, Notch: 20%")
    
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    train_metrics_history = []
    val_metrics_history = []
    
    print("Starting Training")
    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_metrics = validate_one_epoch(model, val_loader, criterion, config.DEVICE)
        
        scheduler.step(val_metrics['total_loss'])
        
        train_metrics_history.append(train_metrics)
        val_metrics_history.append(val_metrics)
        
        # --- MODIFICATION: Update print statements for new metrics ---
        print(f"  TRAIN -> Total: {train_metrics['total_loss']:.4f} | SBP/DBP: {train_metrics['sbp_dbp_loss']:.4f} | Trend: {train_metrics['trend_loss']:.4f} | Notch: {train_metrics['notch_loss']:.4f} | w_mean: {train_metrics['mean_weight']:.3f} | filtered: {train_metrics['filtered_ratio']*100:.1f}%")
        print(f"  VALID -> Total: {val_metrics['total_loss']:.4f} | SBP/DBP: {val_metrics['sbp_dbp_loss']:.4f} | Trend: {val_metrics['trend_loss']:.4f} | Notch: {val_metrics['notch_loss']:.4f} | w_mean: {val_metrics['mean_weight']:.3f} | filtered: {val_metrics['filtered_ratio']*100:.1f}%")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        swanlab.log({
            "epoch": epoch + 1,
            "train/total_loss": train_metrics['total_loss'],
            "train/sbp_dbp_loss": train_metrics['sbp_dbp_loss'],
            "train/trend_loss": train_metrics['trend_loss'],
            "train/notch_loss": train_metrics['notch_loss'],
            "train/mean_weight": train_metrics['mean_weight'],
            "train/filtered_ratio": train_metrics['filtered_ratio'],
            "val/total_loss": val_metrics['total_loss'],
            "val/sbp_dbp_loss": val_metrics['sbp_dbp_loss'],
            "val/trend_loss": val_metrics['trend_loss'],
            "val/notch_loss": val_metrics['notch_loss'],
            "val/mean_weight": val_metrics['mean_weight'],
            "val/filtered_ratio": val_metrics['filtered_ratio'],
            "lr": optimizer.param_groups[0]['lr'],
        })

        val_loss = val_metrics['total_loss']
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(run_models, exist_ok=True)
            model_path = os.path.join(run_models, "best_model.pth")
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'best_val_loss': best_val_loss}, model_path)
            print(f"  Validation loss improved. Saved new best model to {model_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  No improvement. Early stopping counter: {epochs_no_improve}/{config.PATIENCE}")

        # NEW: Save model at every 5th epoch (5, 10, 15, 20, etc.)
        if (epoch + 1) % 5 == 0:
            os.makedirs(run_models, exist_ok=True)
            epoch_model_path = os.path.join(run_models, f"model_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch, 
                'model_state_dict': model.state_dict(), 
                'val_loss': val_loss,
                'train_loss': train_metrics['total_loss']
            }, epoch_model_path)
            print(f"  Saved epoch {epoch+1} model to {epoch_model_path}")
        
        if epochs_no_improve >= config.PATIENCE:
            print(f"No improvement for {config.PATIENCE} consecutive epochs. Stopping early.")
            break
            
    print("Training Finished")
    print(f"Best validation loss achieved: {best_val_loss:.6f}")

    # --- Visualization section ---
    print("Plotting and saving learning curves...")
    try:
        import matplotlib.pyplot as plt
        
        epochs = range(1, len(train_metrics_history) + 1)
        
        # Plot 1: Total Loss Curve (remains the same)
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, [m['total_loss'] for m in train_metrics_history], label='Training Total Loss')
        plt.plot(epochs, [m['total_loss'] for m in val_metrics_history], label='Validation Total Loss')
        plt.title('Training & Validation Total Loss Curve')
        plt.xlabel('Epoch'); plt.ylabel('Loss')
        if val_metrics_history:
            best_epoch = np.argmin([m['total_loss'] for m in val_metrics_history]) + 1
            plt.axvline(best_epoch, color='r', linestyle='--', alpha=0.7, label=f'Best Val Loss (Epoch {best_epoch})')
        plt.legend(); plt.grid(True, alpha=0.3)
        os.makedirs(run_results, exist_ok=True)
        plt.savefig(os.path.join(run_results, "loss_curve.png"), dpi=300)
        plt.close()
        print("Total loss curve saved.")

        # --- MODIFICATION: Plot 2 visualizes the new loss components ---
        plt.figure(figsize=(14, 8))
        
        # SBP/DBP Loss
        plt.plot(epochs, [m['sbp_dbp_loss'] for m in train_metrics_history], label='Train SBP/DBP Loss', color='blue', linestyle='-')
        plt.plot(epochs, [m['sbp_dbp_loss'] for m in val_metrics_history], label='Validation SBP/DBP Loss', color='blue', linestyle='--')
        
        # Trend Loss
        plt.plot(epochs, [m['trend_loss'] for m in train_metrics_history], label='Train Trend Loss', color='green', linestyle='-')
        plt.plot(epochs, [m['trend_loss'] for m in val_metrics_history], label='Validation Trend Loss', color='green', linestyle='--')
        
        # Notch Loss
        plt.plot(epochs, [m['notch_loss'] for m in train_metrics_history], label='Train Notch Loss', color='red', linestyle='-')
        plt.plot(epochs, [m['notch_loss'] for m in val_metrics_history], label='Validation Notch Loss', color='red', linestyle='--')
        
        plt.title('All Loss Components Over Epochs')
        plt.xlabel('Epoch'); plt.ylabel('Loss Value')
        plt.legend(loc='upper right'); plt.grid(True, alpha=0.5)
        plt.yscale('log')
        plt.suptitle('Note: Y-axis is in log scale to show all components clearly', fontsize=10, y=0.92)

        plt.savefig(os.path.join(run_results, "loss_components_curve.png"), dpi=300)
        plt.close()
        print("Loss components curve saved.")

    except ImportError:
        print("Matplotlib not found. Skipping plot generation.")
    except Exception as e:
        print(f"An error occurred during plotting: {e}")

if __name__ == "__main__":
    main()