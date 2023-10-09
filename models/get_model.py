import torch.nn as nn

from models.backbone.resnet import resnet18
from models.modules import DilateEncoder, conv_bn_relu
from models.neck.dla_up import DLAUp
from models.head.yolox_head import YOLOXHead


class res18_c5_dlaup(nn.Module):
    def __init__(self, net_args):
        super(res18_c5_dlaup, self).__init__()
        self.backbone = resnet18()
        channel_reduce_ratio = net_args['parameter']['channel_reduce_ratio']
        channels = [int(64 * channel_reduce_ratio), int(128 * channel_reduce_ratio), int(256 * channel_reduce_ratio),
                    int(512 * channel_reduce_ratio)]
        self.conv1 = conv_bn_relu(64, channels[0], k=1, p=0)
        self.conv2 = conv_bn_relu(128, channels[1], k=1, p=0)
        self.conv3 = conv_bn_relu(256, channels[2], k=1, p=0)
        self.conv4 = DilateEncoder(c1=512, c2=channels[3], dilation_list=[2, 4])
        self.neck = DLAUp(channels, scales=[1, 2, 4, 8])
        self.head = YOLOXHead(net_args['parameter']['heads'], channels[0])

    def forward(self, x):
        outs = []
        feat_list = self.backbone(x)
        x1, x2, x3, x4 = feat_list
        x1 = self.conv1(x1)
        outs.append(x1)
        x2 = self.conv2(x2)
        outs.append(x2)
        x3 = self.conv3(x3)
        outs.append(x3)
        x4 = self.conv4(x4)
        outs.append(x4)
        x = self.neck(outs)

        output = self.head(x)

        return output
