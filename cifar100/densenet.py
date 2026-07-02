import torch
import torch.nn as nn
import torch.nn.functional as F


class _Bottleneck(nn.Module):
    # DenseNet-BC bottleneck: 1x1 -> 3x3, growth_rate = k
    def __init__(self, in_channels, growth_rate, drop_rate=0.0):
        super().__init__()
        inter_channels = 4 * growth_rate

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, padding=1, bias=False)

        self.drop_rate = float(drop_rate)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], dim=1)


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, drop_rate=0.0):
        super().__init__()
        layers = []
        ch = in_channels
        for _ in range(num_layers):
            layers.append(_Bottleneck(ch, growth_rate, drop_rate))
            ch += growth_rate
        self.block = nn.Sequential(*layers)
        self.out_channels = ch

    def forward(self, x):
        return self.block(x)


class _Transition(nn.Module):
    # BN-ReLU-1x1Conv + 2x2 AvgPool
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv(F.relu(self.bn(x), inplace=True))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class DenseNetBC(nn.Module):
    """
    CIFAR DenseNet-BC (Huang et al., 2017)
    depth = 6N + 4 for BC
    growth_rate = k
    compression = theta (0<theta<=1), typical 0.5
    """
    def __init__(self, depth=100, growth_rate=12, compression=0.5, num_classes=100, drop_rate=0.0):
        super().__init__()
        assert (depth - 4) % 6 == 0, "For DenseNet-BC, depth should be 6N+4"
        n = (depth - 4) // 6  # layers per dense block

        self.growth_rate = growth_rate
        self.compression = compression

        # CIFAR stem: 3x3 conv, 2*k channels in the paper
        init_channels = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, init_channels, kernel_size=3, padding=1, bias=False)

        # block 1
        self.block1 = _DenseBlock(n, init_channels, growth_rate, drop_rate)
        ch = self.block1.out_channels
        ch_t = int(ch * compression)
        self.trans1 = _Transition(ch, ch_t)
        ch = ch_t

        # block 2
        self.block2 = _DenseBlock(n, ch, growth_rate, drop_rate)
        ch = self.block2.out_channels
        ch_t = int(ch * compression)
        self.trans2 = _Transition(ch, ch_t)
        ch = ch_t

        # block 3
        self.block3 = _DenseBlock(n, ch, growth_rate, drop_rate)
        ch = self.block3.out_channels

        self.bn = nn.BatchNorm2d(ch)
        self.fc = nn.Linear(ch, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.trans1(self.block1(x))
        x = self.trans2(self.block2(x))
        x = self.block3(x)
        x = F.relu(self.bn(x), inplace=True)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)