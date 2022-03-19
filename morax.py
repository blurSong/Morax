# Morax means the mixing of CMOS and RRAM
# Author: Tao Song.
# Last modification: 0316

import copy
import os
import sys
import argparse
from typing import Container
import re
import numpy as np
import pandas as pd
import subprocess as SP
import multiprocessing as MP
import torch
import morax
import openpyxl
import math


def set_path():
    global home_path, config_path, morax_path, algorithm_path, model_path, result_path
    home_path = os.path.dirname(__file__)
    config_path = os.path.abspath(os.path.join(home_path, "config.cfg"))
    morax_path = os.path.abspath(os.path.join(home_path, "morax"))
    algorithm_path = os.path.abspath(os.path.join(home_path, "algorithm"))
    data_path = os.path.abspath(os.path.join(home_path, 'data'))
    '''
    if os.path.exists(output_path):
        SP.run('rm [{}]{}_dla.csv'.format(dataflow, model), cwd=output_path, shell=True)
        SP.run('rm [{}]{}_rram.csv'.format(dataflow, model), cwd=output_path, shell=True)
        SP.run('rm {}_dla_{}.csv'.format(model, dataflow), cwd=output_path, shell=True)
        SP.run('rm {}_rram_noc*.csv'.format(model), cwd=output_path, shell=True)
    if os.path.exists(model_path):
        SP.run('rm ' + model + '.csv', cwd=model_path, shell=True)
        SP.run('rm ' + model + '_dla_' + dataflow + '.m', cwd=model_path, shell=True)
        SP.run('rm ' + model + '_dla_model.m', cwd=model_path, shell=True)
    '''
    sys.path.append(home_path)
    sys.path.append(data_path)
    sys.path.append(morax_path)
    sys.path.append(algorithm_path)


def set_parser():
    parser = argparse.ArgumentParser(description='Morax simulator parser')
    # parser.add_argument('--type', type=str, default='yxp_os', choices=['ykp_os', 'yxp_os', 'kcp_ws', 'xp_ws', 'rs'])
    parser.add_argument('--scenario', type=str, default='sm', choices=['sm', 'mm', 'cs'])  # single model multi-model context-switching
    parser.add_argument('--model',
                        type=str,
                        default='vgg16',
                        choices=['alexnet', 'vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50', 'resnext50', 'mobilenet_v2', 'shufflenet_v2', 'unet'])
    parser.add_argument('--sec_model',
                        type=str,
                        default='resnet18',
                        choices=['alexnet', 'vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50', 'resnext50', 'mobilenet_v2', 'shufflenet_v2', 'unet'])
    parser.add_argument('--store_trace', action='store_true', default=False)
    parser.add_argument('--use_bn_model', action='store_true', default=False)
    return parser


if __name__ == 'main':
    parser = set_parser()
    args = parser.parse_args()
    set_path()
