import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import config
from data.dataset import PPGDataset
from model.model import UNet1D
from loss.trend_loss import TrendLoss

def calculate_trend_metrics(y_true, y_pred):
    y_true_centered = y_true - np.mean(y_true)
    y_pred_centered = y_pred - np.mean(y_pred)
    
    dot_product = np.sum(y_true_centered * y_pred_centered)
    norm_true = np.sqrt(np.sum(y_true_centered ** 2))
    norm_pred = np.sqrt(np.sum(y_pred_centered ** 2))
    cosine_sim = dot_product / (norm_true * norm_pred + 1e-8)
    
    correlation = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0
    
    if len(y_true) > 1:
        true_grad = np.diff(y_true)
        pred_grad = np.diff(y_pred)
        direction_consistency = np.corrcoef(true_grad, pred_grad)[0, 1] if len(true_grad) > 1 else 0
    else:
        direction_consistency = 0
    
    return {
        'cosine_similarity': cosine_sim,
        'pearson_correlation': correlation,
        'direction_consistency': direction_consistency
    }

def calculate_metrics(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    mse = np.mean((y_true_flat - y_pred_flat) ** 2)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_flat, y_pred_flat)
    
    correlation = np.corrcoef(y_true_flat, y_pred_flat)[0, 1] if len(y_true_flat) > 1 else 0
    
    trend_metrics = calculate_trend_metrics(y_true_flat, y_pred_flat)
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'Correlation': correlation,
        'Cosine_Similarity': trend_metrics['cosine_similarity'],
        'Direction_Consistency': trend_metrics['direction_consistency']
    }

def evaluate_and_plot(model, device, num_samples=5):
    print("Loading test dataset for evaluation...")
    test_dataset_path = os.path.join(config.PROCESSED_DATA_DIR, "test_data.npz")
    if not os.path.exists(test_dataset_path):
        print(f"Error: Test data not found at {test_dataset_path}")
        return
        
    test_dataset = PPGDataset(test_dataset_path)
    if len(test_dataset) == 0:
        print("Test dataset is empty. Cannot perform evaluation.")
        return

    model.eval()
    criterion = TrendLoss()
    
    all_losses = []
    all_metrics = []
    
    sample_indices = np.random.choice(len(test_dataset), size=min(num_samples, len(test_dataset)), replace=False)
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 4 * num_samples), squeeze=False)
    
    with torch.no_grad():
        for i, sample_idx in enumerate(sample_indices):
            sample_data = test_dataset[sample_idx]
            
            if len(sample_data) == 3:
                ppg_sample, bp_true, sensor_mask = sample_data
                ppg_input = ppg_sample.unsqueeze(0).to(device)
                sensor_mask_input = sensor_mask.unsqueeze(0).to(device)
                available_sensors = [config.SENSORS_TO_USE[j] for j in range(len(config.SENSORS_TO_USE)) if sensor_mask[j]]
                sensor_info = f" | Sensors: {len(available_sensors)}/{len(config.SENSORS_TO_USE)} ({available_sensors})"
            else:
                ppg_sample, bp_true = sample_data
                ppg_input = ppg_sample.unsqueeze(0).to(device)
                sensor_mask_input = None
                sensor_info = ""
            
            bp_true_tensor = bp_true.unsqueeze(0).to(device)

            bp_pred = model(ppg_input, sensor_mask_input)
            loss, mse_loss, trend_loss, loss_details = criterion(bp_pred, bp_true_tensor)
            
            all_losses.append({
                'total': loss.item(), 'mse': mse_loss.item(), 'trend': trend_loss.item(),
                'cosine_trend': loss_details['cosine_trend'].item(),
                'corr_trend': loss_details['corr_trend'].item(),
                'grad_trend': loss_details['grad_trend'].item()
            })

            bp_true_np = bp_true.squeeze().cpu().numpy()
            bp_pred_np = bp_pred.squeeze().cpu().numpy()
            
            metrics = calculate_metrics(bp_true_np, bp_pred_np)
            all_metrics.append(metrics)
            
            ax = axes[i, 0]
            ax.plot(bp_true_np, label='Ground Truth BP', color='blue', linewidth=2)
            ax.plot(bp_pred_np, label='Predicted BP', color='red', linestyle='--', linewidth=2)
            ax.set_title(f"Sample #{sample_idx} - MAE: {metrics['MAE']:.3f}, R²: {metrics['R2']:.3f}, Corr: {metrics['Correlation']:.3f}{sensor_info}", fontsize=10)
            ax.legend()
            ax.grid(True, alpha=0.3)

    if not all_metrics:
        print("No metrics were calculated. Evaluation cannot proceed.")
        return

    avg_loss = {key: np.mean([l[key] for l in all_losses]) for key in all_losses[0]}
    avg_metrics = {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0]}
    
    print("\n=== EVALUATION SUMMARY ===")
    print(f"Metrics averaged over {len(sample_indices)} random test samples.")
    print("\nLoss Breakdown:")
    print(f"  Total Loss: {avg_loss['total']:.4f}")
    print(f"  MSE Loss: {avg_loss['mse']:.4f}")
    print(f"  Trend Loss: {avg_loss['trend']:.4f}")
    print(f"    - Grad Trend: {avg_loss['grad_trend']:.4f}")

    print("\nPerformance Metrics:")
    for key, value in avg_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    save_path = os.path.join("results", "prediction_vs_truth_multi_sample.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nEvaluation plot saved to {save_path}")
    plt.close()
    
    metrics_path = os.path.join("results", "evaluation_metrics.txt")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write("血压预测模型评估报告\n")
        f.write("=" * 20 + "\n")
        f.write(f"评估样本数: {len(sample_indices)}\n\n")
        
        f.write("损失函数分解 (平均值):\n")
        for key, value in avg_loss.items():
            f.write(f"  - {key}: {value:.4f}\n")
        
        f.write("\n性能指标 (平均值):\n")
        for key, value in avg_metrics.items():
            f.write(f"  - {key}: {value:.4f}\n")
    
    print(f"Evaluation metrics saved to {metrics_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = UNet1D(in_channels=config.IN_CHANNELS, output_points=config.OUTPUT_POINTS, dropout_rate=0.2).to(device)
    
    model_path = os.path.join("saved_models", "best_model.pth")
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Best model loaded successfully.")
    else:
        print(f"Error: Model file not found at {model_path}")
        return
        
    evaluate_and_plot(model, device, num_samples=5)

if __name__ == "__main__":
    main()