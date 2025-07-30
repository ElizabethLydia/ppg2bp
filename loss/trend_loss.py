import torch
import torch.nn as nn
import torch.nn.functional as F

def pearson_correlation(x, y):
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = torch.sum(xm * ym)
    r_den = torch.sqrt(torch.sum(torch.pow(xm, 2)) * torch.sum(torch.pow(ym, 2)))
    r_val = r_num / (r_den + 1e-8)
    return r_val

class TrendLoss(nn.Module):
    def __init__(self, mse_weight=0.01, trend_weight=0.5, grad_weight=0.49):
        super().__init__()
        self.mse_weight = mse_weight
        self.trend_weight = trend_weight
        self.grad_weight = grad_weight
        self.cosine_similarity = nn.CosineSimilarity(dim=2)

    def forward(self, y_pred, y_true):
        # 1. MSE Loss
        mse_loss = F.mse_loss(y_pred, y_true)

        # 2. Trend Loss Components
        y_pred_centered = y_pred - y_pred.mean(dim=2, keepdim=True)
        y_true_centered = y_true - y_true.mean(dim=2, keepdim=True)
        
        cosine_trend = 1.0 - self.cosine_similarity(y_pred_centered, y_true_centered).mean()
        
        corr_trend = 0.0


        trend_loss = cosine_trend

        # 3. Gradient Loss
        pred_grads = torch.diff(y_pred, dim=2)
        true_grads = torch.diff(y_true, dim=2)
        grad_loss = F.mse_loss(pred_grads, true_grads)

        # 4. Combine
        combined_loss = (self.mse_weight * mse_loss) + \
                        (self.trend_weight * trend_loss) + \
                        (self.grad_weight * grad_loss)
        
        loss_details = {
            'cosine_trend': cosine_trend,
            'corr_trend': torch.tensor(corr_trend), 
            'grad_trend': grad_loss
        }
        
        return combined_loss, mse_loss, trend_loss, loss_details