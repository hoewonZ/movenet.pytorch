from thop import clever_format

import torch
from thop import profile
from lib import init, Data, MoveNet, Task
# from lib.models.movenet_mobilenetv3 import MoveNet
from thop import clever_format
from config import cfg

if __name__ == '__main__':

    input = torch.randn(1, 3, 224, 224)
    model = MoveNet(num_classes=cfg["num_classes"],
                    width_mult=cfg["width_mult"],
                    mode='train')
    macs, params = profile(model, inputs=(input, ))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs)
    print(params)