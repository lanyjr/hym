import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = None
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        res = x if self.shortcut is None else self.shortcut(x)
        out = F.relu(out + res, inplace=True)
        return out


class ResNet3_96(nn.Module):
    """
    CIFAR/SVHN-style small ResNet:
      - stem: 3x3 conv, stride 1
      - stages: [3,3,3] BasicBlocks
      - channels: 96, 192, 384 (stage-wise x2)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_ch = 96

        self.conv1 = nn.Conv2d(3, 96, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(96)

        self.layer1 = self._make_layer(96,  num_blocks=3, stride=1)
        self.layer2 = self._make_layer(192, num_blocks=3, stride=2)
        self.layer3 = self._make_layer(384, num_blocks=3, stride=2)

        self.fc = nn.Linear(384, num_classes)

        self._init_weights()

    def _make_layer(self, out_ch, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_ch, out_ch, stride=s))
            self.in_ch = out_ch
        return nn.Sequential(*layers)

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
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)