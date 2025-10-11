import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedBPAttentionLoss(nn.Module):
    def __init__(self, sbp_dbp_weight=0.5, trend_weight=0.3, notch_weight=0.2, attention_temp=0.1,
                 sbp_dbp_mode: str = None, huber_delta: float = None, mix_alpha: float = None):
        super().__init__()
        self.sbp_dbp_weight = sbp_dbp_weight # SBP/DBP 关键点损失的权重
        self.trend_weight = trend_weight # 整体趋势损失的权重
        self.notch_weight = notch_weight # 二次波/切迹损失的权重
        self.attention_temp = attention_temp # 用于调整注意力掩码锐度的温度参数

        self.mse_loss_fn = nn.MSELoss(reduction='none')
        self.l1_loss_fn = nn.L1Loss(reduction='none')

        # Optional robust modes (default to None -> keep original behavior of MSE)
        self.sbp_dbp_mode = sbp_dbp_mode
        self.huber_delta = huber_delta if huber_delta is not None else 1.0
        self.mix_alpha = mix_alpha if mix_alpha is not None else 0.5

    def _create_attention_mask(self, signal, mode='peak'):
        if mode == 'peak':
            return F.softmax(signal / self.attention_temp, dim=2)
        elif mode == 'trough':
            return F.softmax(-signal / self.attention_temp, dim=2)
        elif mode == 'notch':
            if signal.shape[2] > 2:
                first_derivative = torch.diff(signal, n=1, dim=2)
                second_derivative = torch.diff(first_derivative, n=1, dim=2)
                return F.softmax(F.relu(second_derivative) / self.attention_temp, dim=2)
            else:
                return torch.zeros_like(signal)
        else:
            raise ValueError("Mode must be 'peak', 'trough', or 'notch'")

    def forward(self, y_pred, y_true, sample_weight: torch.Tensor = None):
        if y_pred.dim() == 2:
            y_pred = y_pred.unsqueeze(1)
        if y_true.dim() == 2:
            y_true = y_true.unsqueeze(1)

        # 1. SBP / DBP 关键点损失 (Keypoint Loss)
        sbp_attention = self._create_attention_mask(y_true, mode='peak')
        dbp_attention = self._create_attention_mask(y_true, mode='trough')
        
        keypoint_attention = sbp_attention + dbp_attention
        
        pointwise_mse = self.mse_loss_fn(y_pred, y_true)
        # Robust alternatives for sbp_dbp branch
        mode = (self.sbp_dbp_mode or "mse").lower()
        if mode == "mse":
            keypoint_loss_pw = pointwise_mse
        elif mode == "huber":
            # Smooth L1 / Huber implemented manually for control
            err = torch.abs(y_pred - y_true)
            delta = self.huber_delta
            huber_pw = torch.where(err <= delta, 0.5 * err ** 2, delta * (err - 0.5 * delta))
            keypoint_loss_pw = huber_pw
        elif mode == "mix":
            l1_pw = self.l1_loss_fn(y_pred, y_true)
            keypoint_loss_pw = self.mix_alpha * pointwise_mse + (1.0 - self.mix_alpha) * l1_pw
        else:
            raise ValueError(f"Unsupported sbp_dbp_mode: {self.sbp_dbp_mode}")

        keypoint_loss_pw = keypoint_loss_pw * keypoint_attention
        if sample_weight is not None:
            # broadcast weight to match (B,1,L)
            while sample_weight.dim() < keypoint_loss_pw.dim():
                sample_weight = sample_weight.unsqueeze(-1)
            # normalize by mean weight to keep loss scale
            sbp_dbp_loss = (keypoint_loss_pw * sample_weight).mean() / (sample_weight.mean() + 1e-8)
        else:
            sbp_dbp_loss = keypoint_loss_pw.mean()

        # 2. 整体趋势损失 (Trend Loss)
        y_pred_centered = y_pred - y_pred.mean(dim=2, keepdim=True)
        y_true_centered = y_true - y_true.mean(dim=2, keepdim=True)
        
        trend_pw = 1.0 - F.cosine_similarity(y_pred_centered, y_true_centered, dim=2)
        if sample_weight is not None:
            trend_loss = (trend_pw * sample_weight.squeeze()).mean() / (sample_weight.mean() + 1e-8)
        else:
            trend_loss = trend_pw.mean()

        # 3. 二次波/切迹损失 (Dicrotic Notch Loss)
        notch_attention = self._create_attention_mask(y_true, mode='notch')
        
        pred_padded_for_notch = y_pred[:, :, 2:]
        true_padded_for_notch = y_true[:, :, 2:]
        pointwise_mse_notch = self.mse_loss_fn(pred_padded_for_notch, true_padded_for_notch)
        
        notch_pw = pointwise_mse_notch * notch_attention
        if sample_weight is not None:
            notch_loss = (notch_pw * sample_weight.unsqueeze(-1)).mean() / (sample_weight.mean() + 1e-8)
        else:
            notch_loss = notch_pw.mean()
        
        # 4. 组合所有损失
        combined_loss = (self.sbp_dbp_weight * sbp_dbp_loss) + \
                        (self.trend_weight * trend_loss) + \
                        (self.notch_weight * notch_loss)

        loss_details = {
            'total_loss': combined_loss.item(),
            'sbp_dbp_loss': sbp_dbp_loss.item(),
            'trend_loss': trend_loss.item(),
            'notch_loss': notch_loss.item()
        }
        return combined_loss, loss_details