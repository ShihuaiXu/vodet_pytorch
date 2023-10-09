import torch.nn as nn
from models.modules import fill_fc_weights


class YOLOXHead(nn.Module):
    def __init__(self, heads, head_conv):
        super(YOLOXHead, self).__init__()
        self.heads = heads
        for head, classes in self.heads.items():
            setattr(self, head + '_fc', nn.Sequential(
                      nn.Conv2d(head_conv, 64,
                        kernel_size=3, padding=1, bias=True),
                      nn.BatchNorm2d(64),
                      nn.ReLU(inplace=True),
                      nn.Conv2d(64, classes,
                        kernel_size=1, stride=1,
                        padding=0, bias=True)))
            if 'hm' in head and head != 'hm_hp_offset':
                getattr(self, head + '_fc')[-1].bias.data.fill_(-2.19)
            else:
                fill_fc_weights(getattr(self, head + '_fc'))

    def forward(self, x):
        output = {}
        for head in self.heads:
            out = getattr(self, head + '_fc')(x)
            output[head] = out

        return output
