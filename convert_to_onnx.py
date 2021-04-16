from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import time
from config import cfg_plate
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.basemodel import BaseModel

from utils.box_utils import decode, decode_landm
from utils.timer import Timer
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('-m', '--trained_model', default='weights/LP_detect/LP_detect_150.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def get_image_path(path):
    out = []
    for path, subdirs, files in os.walk(path):
        for name in files:
            if name[-4:] == '.jpg':
                out.append(os.path.join(path, name))
    return out


if __name__ == '__main__':
    torch.set_grad_enabled(False)

    cfg = None
    net = None
    if args.trained_model is not None:
        cfg = cfg_plate
        net = BaseModel(cfg=cfg, phase='test')
    else:
        print("Don't support network!")
        exit(0)

    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    ##################export###############
    output_onnx = 'Retina_Plate_dynamix_size.onnx'
    print("==> Exporting model to ONNX format at '{}'".format(output_onnx))
    input_names = ["input0"]
    output_names = ["loc0", "conf0", "landmark0"]
    inputs = torch.randn(1, 3, 480, 850)
    torch_out = torch.onnx._export(net,
                                   inputs,
                                   output_onnx,
                                   verbose=True,
                                   input_names=input_names,
                                   output_names=output_names,
                                   example_outputs=True,  # to show sample output dimension
                                   keep_initializers_as_inputs=True,  # to avoid error _Map_base::at
                                   # opset_version=11, # need to change to 11, to deal with tensorflow fix_size input
                                   dynamic_axes={
                                       "input0": [2, 3],
                                       "loc0": [1, 2],
                                       "conf0": [1, 2],
                                       "landmark0": [1, 2]
                                   }
                                   )