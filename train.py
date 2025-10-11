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
        
        # --- MODIFICATION: Unpack new loss and details dictionary ---
        loss, loss_details = criterion(predictions, bp)
        
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
    return {
        'total_loss': running_total_loss / dataset_size,
        'sbp_dbp_loss': running_sbp_dbp_loss / dataset_size,
        'trend_loss': running_trend_loss / dataset_size,
        'notch_loss': running_notch_loss / dataset_size
    }

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()

    # --- MODIFICATION: Update running totals for new loss components ---
    running_total_loss = 0.0
    running_sbp_dbp_loss = 0.0
    running_trend_loss = 0.0
    running_notch_loss = 0.0
    
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
            
            # --- MODIFICATION: Unpack new loss and details dictionary ---
            loss, loss_details = criterion(predictions, bp)
            
            if not torch.isnan(loss) and not torch.isinf(loss):
                # --- MODIFICATION: Accumulate new loss components ---
                batch_size = ppg.size(0)
                running_total_loss += loss.item() * batch_size
                running_sbp_dbp_loss += loss_details['sbp_dbp_loss'] * batch_size
                running_trend_loss += loss_details['trend_loss'] * batch_size
                running_notch_loss += loss_details['notch_loss'] * batch_size

    dataset_size = len(dataloader.dataset)
    
    # --- MODIFICATION: Return dictionary with new metrics ---
    return {
        'total_loss': running_total_loss / dataset_size,
        'sbp_dbp_loss': running_sbp_dbp_loss / dataset_size,
        'trend_loss': running_trend_loss / dataset_size,
        'notch_loss': running_notch_loss / dataset_size
    }

def main():
    # Phase 0: enforce reproducibility
    set_seed(getattr(config, "SEED", 42))
    swanlab.init(
    project="ppg2bp-train",
    name="unet1d-bp-loss-seg1*",
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
    print("Loading datasets...")
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
        print(f"  TRAIN -> Total: {train_metrics['total_loss']:.4f} | SBP/DBP: {train_metrics['sbp_dbp_loss']:.4f} | Trend: {train_metrics['trend_loss']:.4f} | Notch: {train_metrics['notch_loss']:.4f}")
        print(f"  VALID -> Total: {val_metrics['total_loss']:.4f} | SBP/DBP: {val_metrics['sbp_dbp_loss']:.4f} | Trend: {val_metrics['trend_loss']:.4f} | Notch: {val_metrics['notch_loss']:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        swanlab.log({
            "epoch": epoch + 1,
            "train/total_loss": train_metrics['total_loss'],
            "train/sbp_dbp_loss": train_metrics['sbp_dbp_loss'],
            "train/trend_loss": train_metrics['trend_loss'],
            "train/notch_loss": train_metrics['notch_loss'],
            "val/total_loss": val_metrics['total_loss'],
            "val/sbp_dbp_loss": val_metrics['sbp_dbp_loss'],
            "val/trend_loss": val_metrics['trend_loss'],
            "val/notch_loss": val_metrics['notch_loss'],
            "lr": optimizer.param_groups[0]['lr'],
        })

        val_loss = val_metrics['total_loss']
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(config.SAVED_MODELS_DIR, exist_ok=True)
            model_path = os.path.join(config.SAVED_MODELS_DIR, "best_model.pth")
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'best_val_loss': best_val_loss}, model_path)
            print(f"  Validation loss improved. Saved new best model to {model_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  No improvement. Early stopping counter: {epochs_no_improve}/{config.PATIENCE}")

        # NEW: Save model at every 5th epoch (5, 10, 15, 20, etc.)
        if (epoch + 1) % 5 == 0:
            os.makedirs(config.SAVED_MODELS_DIR, exist_ok=True)
            epoch_model_path = os.path.join(config.SAVED_MODELS_DIR, f"model_epoch_{epoch+1}.pth")
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
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        plt.savefig(os.path.join(config.RESULTS_DIR, "loss_curve.png"), dpi=300)
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

        plt.savefig(os.path.join(config.RESULTS_DIR, "loss_components_curve.png"), dpi=300)
        plt.close()
        print("Loss components curve saved.")

    except ImportError:
        print("Matplotlib not found. Skipping plot generation.")
    except Exception as e:
        print(f"An error occurred during plotting: {e}")

if __name__ == "__main__":
    main()