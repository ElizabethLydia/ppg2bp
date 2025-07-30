import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os
import numpy as np

import config
from data.dataset import PPGDataset
from model.model import UNet1D
from loss.trend_loss import TrendLoss

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_mse_loss = 0.0
    running_trend_loss = 0.0
    running_cosine_trend = 0.0
    running_corr_trend = 0.0
    running_grad_trend = 0.0
    
    for batch_data in tqdm(dataloader, desc="Training"):
        if len(batch_data) == 3:  # 新格式：ppg, bp, mask
            ppg, bp, sensor_mask = batch_data
            ppg, bp, sensor_mask = ppg.to(device), bp.to(device), sensor_mask.to(device)
        else:  # 旧格式：ppg, bp
            ppg, bp = batch_data
            ppg, bp = ppg.to(device), bp.to(device)
            sensor_mask = None
        
        optimizer.zero_grad()
        
        predictions = model(ppg, sensor_mask)
        
        loss, mse_loss, trend_loss, loss_details = criterion(predictions, bp)
        
        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: NaN or Inf loss detected, skipping batch")
            continue
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        batch_size = ppg.size(0)
        running_loss += loss.item() * batch_size
        running_mse_loss += mse_loss.item() * batch_size
        running_trend_loss += trend_loss.item() * batch_size
        running_cosine_trend += loss_details['cosine_trend'].item() * batch_size
        running_corr_trend += loss_details['corr_trend'].item() * batch_size
        running_grad_trend += loss_details['grad_trend'].item() * batch_size
        
    dataset_size = len(dataloader.dataset)
    epoch_loss = running_loss / dataset_size
    epoch_mse_loss = running_mse_loss / dataset_size
    epoch_trend_loss = running_trend_loss / dataset_size
    epoch_cosine_trend = running_cosine_trend / dataset_size
    epoch_corr_trend = running_corr_trend / dataset_size
    epoch_grad_trend = running_grad_trend / dataset_size
    
    return {
        'total_loss': epoch_loss,
        'mse_loss': epoch_mse_loss,
        'trend_loss': epoch_trend_loss,
        'cosine_trend': epoch_cosine_trend,
        'corr_trend': epoch_corr_trend,
        'grad_trend': epoch_grad_trend
    }

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_mse_loss = 0.0
    running_trend_loss = 0.0
    running_cosine_trend = 0.0
    running_corr_trend = 0.0
    running_grad_trend = 0.0
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Validating"):
            if len(batch_data) == 3:  # 新格式：ppg, bp, mask
                ppg, bp, sensor_mask = batch_data
                ppg, bp, sensor_mask = ppg.to(device), bp.to(device), sensor_mask.to(device)
            else:  # 旧格式：ppg, bp
                ppg, bp = batch_data
                ppg, bp = ppg.to(device), bp.to(device)
                sensor_mask = None
                
            predictions = model(ppg, sensor_mask)
            loss, mse_loss, trend_loss, loss_details = criterion(predictions, bp)
            
            if not torch.isnan(loss) and not torch.isinf(loss):
                batch_size = ppg.size(0)
                running_loss += loss.item() * batch_size
                running_mse_loss += mse_loss.item() * batch_size
                running_trend_loss += trend_loss.item() * batch_size
                running_cosine_trend += loss_details['cosine_trend'].item() * batch_size
                running_corr_trend += loss_details['corr_trend'].item() * batch_size
                running_grad_trend += loss_details['grad_trend'].item() * batch_size
    
    dataset_size = len(dataloader.dataset)
    epoch_loss = running_loss / dataset_size
    epoch_mse_loss = running_mse_loss / dataset_size
    epoch_trend_loss = running_trend_loss / dataset_size
    epoch_cosine_trend = running_cosine_trend / dataset_size
    epoch_corr_trend = running_corr_trend / dataset_size
    epoch_grad_trend = running_grad_trend / dataset_size

    return {
        'total_loss': epoch_loss,
        'mse_loss': epoch_mse_loss,
        'trend_loss': epoch_trend_loss,
        'cosine_trend': epoch_cosine_trend,
        'corr_trend': epoch_corr_trend,
        'grad_trend': epoch_grad_trend
    }

def main():
    print("Loading datasets...")
    train_dataset = PPGDataset(os.path.join(config.PROCESSED_DATA_DIR, "train_data.npz"))
    val_dataset = PPGDataset(os.path.join(config.PROCESSED_DATA_DIR, "validation_data.npz"))
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Error: Train or validation dataset is empty.")
        return

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, 
                             num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, 
                           num_workers=4, pin_memory=True, drop_last=False)
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    print(f"Initializing model on device: {config.DEVICE}")
    model = UNet1D(in_channels=config.IN_CHANNELS, output_points=config.OUTPUT_POINTS, dropout_rate=0.4).to(config.DEVICE)
    
    criterion = TrendLoss(mse_weight=0.01, trend_weight=0.5, grad_weight=0.49)
    print("Using enhanced TrendLoss - MSE:30% + Trend:50% + Grad:20%")
    
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    
    print("Starting Training")
    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_metrics = validate_one_epoch(model, val_loader, criterion, config.DEVICE)
        
        scheduler.step(val_metrics['total_loss'])
        
        train_losses.append(train_metrics['total_loss'])
        val_losses.append(val_metrics['total_loss'])
        
        print(f"  TRAIN -> Total: {train_metrics['total_loss']:.4f} | MSE: {train_metrics['mse_loss']:.4f} | Trend: {train_metrics['trend_loss']:.4f}")
        print(f"           Cosine: {train_metrics['cosine_trend']:.4f} | Corr: {train_metrics['corr_trend']:.4f} | Grad: {train_metrics['grad_trend']:.4f}")
        print(f"  VALID -> Total: {val_metrics['total_loss']:.4f} | MSE: {val_metrics['mse_loss']:.4f} | Trend: {val_metrics['trend_loss']:.4f}")
        print(f"           Cosine: {val_metrics['cosine_trend']:.4f} | Corr: {val_metrics['corr_trend']:.4f} | Grad: {val_metrics['grad_trend']:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        val_loss = val_metrics['total_loss']
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("saved_models", exist_ok=True)
            model_path = os.path.join("saved_models", "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }, model_path)
            print(f"  Validation loss improved. Saved new best model to {model_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  No improvement. Early stopping counter: {epochs_no_improve}/{config.PATIENCE}")
        
        if epochs_no_improve >= config.PATIENCE:
            print(f"No improvement for {config.PATIENCE} consecutive epochs. Stopping early.")
            break
            
    print("Training Finished")
    print(f"Best validation loss achieved: {best_val_loss:.6f}")

    print("Plotting and saving loss curve...")
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.plot(train_losses, label='Training Loss', linewidth=2)
        plt.plot(val_losses, label='Validation Loss', linewidth=2)
        plt.title('Training & Validation Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if val_losses:
            best_epoch = np.argmin(val_losses)
            plt.axvline(best_epoch, color='r', linestyle='--', alpha=0.7, 
                       label=f'Best Val Loss (Epoch {best_epoch+1})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        os.makedirs("results", exist_ok=True)
        plot_path = os.path.join("results", "loss_curve.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Loss curve saved to {plot_path}")
        plt.close()
    except ImportError:
        print("Matplotlib not found. Skipping plot generation.")
    except Exception as e:
        print(f"Error during plotting: {e}")

if __name__ == "__main__":
    main()