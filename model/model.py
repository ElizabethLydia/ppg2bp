import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

class AttentionBlock(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(AttentionBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class UNet1D(nn.Module):
    def __init__(self, in_channels, output_points, dropout_rate=0.2):
        super(UNet1D, self).__init__()
        self.output_points = output_points
        self.in_channels = in_channels
        
        # 输入适配层 - 处理可变数量的传感器
        self.input_adapter = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=1, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )
        
        self.initial_conv = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=15, stride=1, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        self.encoder1 = nn.Sequential(
            ResidualBlock(32, 64, stride=2, dropout_rate=dropout_rate),
            AttentionBlock(64)
        )
        
        self.encoder2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2, dropout_rate=dropout_rate),
            AttentionBlock(128)
        )
        
        self.encoder3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2, dropout_rate=dropout_rate),
            AttentionBlock(256)
        )
        
        self.bottleneck = nn.Sequential(
            ResidualBlock(256, 512, stride=2, dropout_rate=dropout_rate),
            AttentionBlock(512),
            ResidualBlock(512, 512, stride=1, dropout_rate=dropout_rate)
        )
        
        self.decoder3 = nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1)
        self.decode_block3 = ResidualBlock(512, 256, dropout_rate=dropout_rate)
        
        self.decoder2 = nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1)
        self.decode_block2 = ResidualBlock(256, 128, dropout_rate=dropout_rate)
        
        self.decoder1 = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1)
        self.decode_block1 = ResidualBlock(128, 64, dropout_rate=dropout_rate)
        
        self.final_decoder = nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)
        self.final_conv = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 1, kernel_size=1)
        )

    def forward(self, x, sensor_mask=None):
        if torch.isnan(x).any():
            x = torch.nan_to_num(x)
        
        # 处理传感器掩码
        if sensor_mask is not None:
            # 将缺失传感器的数据置零
            sensor_mask_expanded = sensor_mask.unsqueeze(-1).float()  # [B, C, 1]
            x = x * sensor_mask_expanded
        
        # 输入适配
        x = self.input_adapter(x)
        x1 = self.initial_conv(x)
        
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        
        bottleneck = self.bottleneck(x4)
        
        up3 = self.decoder3(bottleneck)
        up3 = self._match_size(up3, x4)
        up3 = torch.cat([up3, x4], dim=1)
        up3 = self.decode_block3(up3)
        
        up2 = self.decoder2(up3)
        up2 = self._match_size(up2, x3)
        up2 = torch.cat([up2, x3], dim=1)
        up2 = self.decode_block2(up2)
        
        up1 = self.decoder1(up2)
        up1 = self._match_size(up1, x2)
        up1 = torch.cat([up1, x2], dim=1)
        up1 = self.decode_block1(up1)
        
        final_up = self.final_decoder(up1)
        final_up = self._match_size(final_up, x1)
        final_up = torch.cat([final_up, x1], dim=1)
        
        output = self.final_conv(final_up)
        
        if output.size(-1) != self.output_points:
            output = F.interpolate(output, size=self.output_points, mode='linear', align_corners=False)
        
        return output

    def _match_size(self, x, target):
        if x.size(-1) != target.size(-1):
            x = F.interpolate(x, size=target.size(-1), mode='linear', align_corners=False)
        return x