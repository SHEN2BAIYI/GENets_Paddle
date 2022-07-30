import paddle
import torch
import numpy as np
from x2paddle.convert import pytorch2paddle

from model.genet_paddle import *


def transfer():
    input_fp = '../weight/pytorch/GENet_light.pth'
    output_fp = '../weight/paddle/GENet_light.pdparams'
    torch_dict = torch.load(input_fp)
    print(torch_dict)
    paddle_dict = {}
    fc_names = [
        'classifier.1.weight', 'classifier.4.weright',
    ]

    for key in torch_dict:
        weight = torch_dict[key]
        print(type(weight))


transfer()
