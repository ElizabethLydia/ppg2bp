import sys
import os

# Ensure the repo root (folder containing config.py) is on sys.path
repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, repo_root)

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import config
from data.dataset import PPGDataset
from model.model import UNet1D
from loss.trend_loss import CombinedBPAttentionLoss
import glob
import argparse

try:
    import swanlab
except Exception:
    class _SwanStub:
        def init(self, **kwargs):
            print("[swanlab stub] init", kwargs)
        def log(self, data):
            pass
        class Image:
            def __init__(self, path):
                self.path = path
    swanlab = _SwanStub()


def calculate_metrics(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    mse = np.mean((y_true_flat - y_pred_flat) ** 2)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_flat, y_pred_flat)
    correlation = np.corrcoef(y_true_flat, y_pred_flat)[0, 1] if len(y_true_flat) > 1 else 0
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'Correlation': correlation,
    }


def evaluate_model_on_dataset(model, dataset, device):
    """Evaluate a single model on the entire dataset and return average metrics"""
    model.eval()
    all_metrics = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            sample_data = dataset[i]
            
            if len(sample_data) == 3:
                ppg_sample, bp_true, sensor_mask = sample_data
                ppg_input = ppg_sample.unsqueeze(0).to(device)
                sensor_mask_input = sensor_mask.unsqueeze(0).to(device)
            else:
                ppg_sample, bp_true = sample_data
                ppg_input = ppg_sample.unsqueeze(0).to(device)
                sensor_mask_input = None
            
            bp_pred = model(ppg_input, sensor_mask_input)
            
            bp_true_np = bp_true.squeeze().cpu().numpy()
            bp_pred_np = bp_pred.squeeze().cpu().numpy()
            
            metrics = calculate_metrics(bp_true_np, bp_pred_np)
            all_metrics.append(metrics)
    
    # Calculate average metrics
    avg_metrics = {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0]}
    return avg_metrics


def evaluate_and_plot(model, device, model_name, results_dir, num_samples=5):
    print(f"Loading test dataset for evaluation of {model_name}...")
    test_dataset_path = os.path.join(config.PROCESSED_DATA_DIR, "test_data.npz")
    if not os.path.exists(test_dataset_path):
        print(f"Error: Test data not found at {test_dataset_path}")
        return None
        
    test_dataset = PPGDataset(test_dataset_path)
    if len(test_dataset) == 0:
        print("Test dataset is empty. Cannot perform evaluation.")
        return None

    model.eval()
    
    all_metrics = []
    
    if len(test_dataset) > num_samples:
        sample_indices = np.random.choice(len(test_dataset), size=num_samples, replace=False)
    else:
        sample_indices = np.arange(len(test_dataset))
    
    fig, axes = plt.subplots(len(sample_indices), 1, figsize=(15, 4 * len(sample_indices)), squeeze=False)
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            sample_data = test_dataset[i]
            
            if len(sample_data) == 3:
                ppg_sample, bp_true, sensor_mask = sample_data
                ppg_input = ppg_sample.unsqueeze(0).to(device)
                sensor_mask_input = sensor_mask.unsqueeze(0).to(device)
            else:
                ppg_sample, bp_true = sample_data
                ppg_input = ppg_sample.unsqueeze(0).to(device)
                sensor_mask_input = None
            
            bp_pred = model(ppg_input, sensor_mask_input)
            
            bp_true_np = bp_true.squeeze().cpu().numpy()
            bp_pred_np = bp_pred.squeeze().cpu().numpy()
            
            metrics = calculate_metrics(bp_true_np, bp_pred_np)
            all_metrics.append(metrics)
            
            if i in sample_indices:
                plot_idx = np.where(sample_indices == i)[0][0]
                ax = axes[plot_idx, 0]
                ax.plot(bp_true_np, label='Ground Truth BP', color='blue', linewidth=2)
                ax.plot(bp_pred_np, label='Predicted BP', color='red', linestyle='--', linewidth=2)
                ax.set_title(f"{model_name} - Sample #{i} - MAE: {metrics['MAE']:.3f}, R²: {metrics['R2']:.3f}, Corr: {metrics['Correlation']:.3f}", fontsize=10)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
    if not all_metrics:
        print("No metrics were calculated. Evaluation cannot proceed.")
        return None

    avg_metrics = {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0]}
    
    print(f"\n=== EVALUATION SUMMARY for {model_name} ===")
    print(f"Metrics averaged over the entire test set ({len(test_dataset)} samples).")
    print("\nPerformance Metrics:")
    for key, value in avg_metrics.items():
        print(f"  {key}: {value:.4f}")

    plt.tight_layout()
    os.makedirs(results_dir, exist_ok=True)
    safe_model_name = model_name.replace("/", "_").replace("\\", "_")
    save_path = os.path.join(results_dir, f"evaluation_{safe_model_name}_plots.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nEvaluation plot saved to {save_path}")
    plt.close()
    
    return avg_metrics, save_path


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # CLI args
    parser = argparse.ArgumentParser(description="Evaluate saved PPG2BP models")
    parser.add_argument("--models_dir", type=str, default=config.SAVED_MODELS_DIR,
                        help="Directory containing model .pth files (default: config.SAVED_MODELS_DIR)")
    parser.add_argument("--results_dir", type=str, default=config.RESULTS_DIR,
                        help="Directory to write evaluation plots and reports (default: config.RESULTS_DIR)")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples to visualize")
    args = parser.parse_args()

    swanlab.init(
        project="ppg2bp-evaluation",
        name="unet1d-bp-seg1-multi-epoch"
    )
    
    # NEW: Find all available model files
    model_files = []
    
    # Add best model (use config path)
    best_model_path = os.path.join(args.models_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        model_files.append(("best_model", best_model_path))
    
    # Add epoch models (5, 10, 15, etc.)
    epoch_model_pattern = os.path.join(args.models_dir, "model_epoch_*.pth")
    epoch_models = glob.glob(epoch_model_pattern)
    
    for epoch_model_path in sorted(epoch_models):
        # Extract epoch number from filename
        filename = os.path.basename(epoch_model_path)
        epoch_num = filename.replace("model_epoch_", "").replace(".pth", "")
        model_name = f"epoch_{epoch_num}"
        model_files.append((model_name, epoch_model_path))
    
    if not model_files:
        print("No model files found in saved_models directory!")
        return
    
    print(f"Found {len(model_files)} models to evaluate:")
    for model_name, model_path in model_files:
        print(f"  - {model_name}: {model_path}")
    
    # NEW: Evaluate each model and collect results
    all_results = {}
    
    for model_name, model_path in model_files:
        print(f"\n{'='*50}")
        print(f"Evaluating {model_name}")
        print(f"{'='*50}")
        
        # Load model
        model = UNet1D(in_channels=config.IN_CHANNELS, output_points=config.OUTPUT_POINTS, dropout_rate=0.2).to(device)
        
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded successfully from {model_path}")
            
            # Get epoch info if available
            if 'epoch' in checkpoint:
                print(f"Model was saved at epoch: {checkpoint['epoch'] + 1}")
            if 'val_loss' in checkpoint:
                print(f"Validation loss at save time: {checkpoint['val_loss']:.4f}")
            elif 'best_val_loss' in checkpoint:
                print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
                
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            continue
        
        # Evaluate model
        result = evaluate_and_plot(model, device, model_name, results_dir=args.results_dir, num_samples=args.num_samples)
        if result is not None:
            avg_metrics, plot_path = result
            all_results[model_name] = avg_metrics
            
            # Log to swanlab with model-specific prefix
            swanlab_metrics = {f"{model_name}/{key}": value for key, value in avg_metrics.items()}
            swanlab.log(swanlab_metrics)
            swanlab.log({f"{model_name}/plot": swanlab.Image(plot_path)})
    
    # NEW: Create comparison plot and summary
    if len(all_results) > 1:
        print(f"\n{'='*50}")
        print("COMPARISON SUMMARY")
        print(f"{'='*50}")
        
        # Create comparison table
        metrics_to_compare = ['MAE', 'RMSE', 'R2', 'Correlation']
        
        print(f"{'Model':<15} | {'MAE':<8} | {'RMSE':<8} | {'R2':<8} | {'Corr':<8}")
        print("-" * 60)
        
        for model_name, metrics in all_results.items():
            print(f"{model_name:<15} | {metrics['MAE']:<8.4f} | {metrics['RMSE']:<8.4f} | {metrics['R2']:<8.4f} | {metrics['Correlation']:<8.4f}")
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison Across Epochs', fontsize=16)
        
        models = list(all_results.keys())
        
        # Extract epoch numbers for x-axis (for epoch models)
        epoch_numbers = []
        epoch_models = []
        best_model_metrics = None
        
        for model_name in models:
            if model_name == "best_model":
                best_model_metrics = all_results[model_name]
            elif model_name.startswith("epoch_"):
                epoch_num = int(model_name.replace("epoch_", ""))
                epoch_numbers.append(epoch_num)
                epoch_models.append(model_name)
        
        # Sort by epoch number
        if epoch_numbers:
            sorted_pairs = sorted(zip(epoch_numbers, epoch_models))
            epoch_numbers, epoch_models = zip(*sorted_pairs)
        
        for idx, metric in enumerate(metrics_to_compare):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            if epoch_numbers:
                # Plot epoch models
                epoch_values = [all_results[model][metric] for model in epoch_models]
                ax.plot(epoch_numbers, epoch_values, 'b-o', label='Epoch Models', linewidth=2, markersize=6)
                
                # Add best model as horizontal line if available
                if best_model_metrics:
                    ax.axhline(y=best_model_metrics[metric], color='red', linestyle='--', 
                              linewidth=2, label='Best Model', alpha=0.8)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} Across Epochs')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        comparison_plot_path = os.path.join(args.results_dir, "model_comparison_across_epochs.png")
        plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nComparison plot saved to {comparison_plot_path}")
        
        swanlab.log({"comparison/plot": swanlab.Image(comparison_plot_path)})
        
        # Save detailed comparison report
        report_path = os.path.join(args.results_dir, "multi_epoch_evaluation_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Multi-Epoch Model Evaluation Report\n")
            f.write("=" * 40 + "\n\n")
            
            f.write(f"Evaluated {len(all_results)} models:\n")
            for model_name in all_results.keys():
                f.write(f"  - {model_name}\n")
            f.write("\n")
            
            f.write("Performance Metrics Comparison:\n")
            f.write(f"{'Model':<15} | {'MAE':<8} | {'RMSE':<8} | {'R2':<8} | {'Corr':<8}\n")
            f.write("-" * 60 + "\n")
            
            for model_name, metrics in all_results.items():
                f.write(f"{model_name:<15} | {metrics['MAE']:<8.4f} | {metrics['RMSE']:<8.4f} | {metrics['R2']:<8.4f} | {metrics['Correlation']:<8.4f}\n")
            
            f.write("\n\nDetailed Analysis:\n")
            
            # Find best performing model for each metric
            for metric in metrics_to_compare:
                if metric in ['MAE', 'MSE', 'RMSE']:  # Lower is better
                    best_model = min(all_results.items(), key=lambda x: x[1][metric])
                    f.write(f"Best {metric}: {best_model[0]} ({best_model[1][metric]:.4f})\n")
                else:  # Higher is better
                    best_model = max(all_results.items(), key=lambda x: x[1][metric])
                    f.write(f"Best {metric}: {best_model[0]} ({best_model[1][metric]:.4f})\n")
        
    print(f"Detailed comparison report saved to {report_path}")
    
    print(f"\nEvaluation complete! Evaluated {len(all_results)} models.")


if __name__ == "__main__":
    main()