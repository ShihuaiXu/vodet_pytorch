import torch.nn as nn


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class conv_bn_relu(nn.Module):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1, momentum=0.1):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, momentum=momentum)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c, d=1, e=0.25):
        super(Bottleneck, self).__init__()
        c_ = int(c * e)
        self.branch = nn.Sequential(
            conv_bn_relu(c, c_, k=1),
            conv_bn_relu(c_, c_, k=3, p=d, d=d),
            conv_bn_relu(c_, c, k=1)
        )

    def forward(self, x):
        out = x + self.branch(x)
        return out


class DilateEncoder(nn.Module):
    """ DilateEncoder """

    def __init__(self, c1, c2, dilation_list=[2, 4, 6, 8]):
        super(DilateEncoder, self).__init__()

        # projector改变通道维度
        self.projector = nn.Sequential(
            conv_bn_relu(c1, c2, k=1),
            conv_bn_relu(c2, c2, k=3, p=1)
        )

        encoders = []
        for d in dilation_list:
            encoders.append(Bottleneck(c2, d=d))
        self.encoders = nn.Sequential(*encoders)

        self._init_weight()

    def _init_weight(self):
        for m in self.projector:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.encoders.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.projector(x)
        x = self.encoders(x)
        return x
