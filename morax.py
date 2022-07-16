# Morax means the mixing of CMOS and RRAM
# Author: Tao Song.
# Last modification: 0316

from ast import Return
import os
import sys
import argparse
from typing import Container
import re
from unittest import result
from docutils import DataError
import numpy as np
import pandas as pd
import subprocess as SP
import multiprocessing as MP
import openpyxl
import math

from morax.frontend import api, csvparser
from morax.model import model
from morax.hardware import chip
from morax.system import memonitor, query, mapper

import algorithm.online as OL
import algorithm.offline as OFL


def set_path():
    global home_path, morax_path, algorithm_path, data_path
    home_path = os.path.dirname(__file__)
    morax_path = os.path.abspath(os.path.join(home_path, "morax"))
    algorithm_path = os.path.abspath(os.path.join(home_path, "algorithm"))
    data_path = os.path.abspath(os.path.join(home_path, "data"))
    """
    if os.path.exists(output_path):
        SP.run('rm [{}]{}_dla.csv'.format(dataflow, model), cwd=output_path, shell=True)
    if os.path.exists(model_path):
        SP.run('rm ' + model + '.csv', cwd=model_path, shell=True)
    """
    sys.path.append(home_path)
    sys.path.append(data_path)
    sys.path.append(morax_path)
    sys.path.append(algorithm_path)


def set_parser():
    parser = argparse.ArgumentParser(description="Morax simulator parser")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument(
        "--scenario", type=str, default="sm", choices=["sm", "mm", "cs"]
    )  # single model multi-model context-switching
    # parser.add_argument("--strategy", type=int, default="2", choices=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--model",
        type=str,
        default="vgg16",
        choices=[
            "alexnet",
            "vgg16",
            "vgg19",
            "resnet18",
            "resnet34",
            "resnet50",
            "resnext50",
            "mobilenet_v2",
            "shufflenet_v2",
            "unet",
        ],
    )
    parser.add_argument(
        "--sec_model",
        type=str,
        default="resnet18",
        choices=[
            "alexnet",
            "vgg16",
            "vgg19",
            "resnet18",
            "resnet34",
            "resnet50",
            "resnext50",
            "mobilenet_v2",
            "shufflenet_v2",
            "unet",
        ],
    )
    parser.add_argument("--store_trace", action="store_true", default=False)
    parser.add_argument("--use_non_normal_model", action="store_true", default=False)

    return parser


def get_modeltype(_modelname):
    if _modelname in model.CNNModelList:
        model_type = model.ModelType.CNN
    elif _modelname in model.MLPModelList:
        model_type = model.ModelType.MLP
    elif _modelname in model.AttentionModelList:
        model_type = model.ModelType.MHATTENTION
    return model_type


if __name__ == "__main__":
    parser = set_parser()
    args = parser.parse_args()
    set_path()

    # csv pre-process
    model_path = os.path.abspath(os.path.join(data_path, "model"))
    result_path = os.path.abspath(os.path.join(data_path, "result_path"))
    model_type = get_modeltype(args.model)
    if model_type == model.ModelType.CNN:
        LUTEN_ = True
        if args.use_non_normal_model:
            csvparser.remove_bn_to_csv(model_path, args.model)
    elif model_type == model.ModelType.MLP:
        LUTEN_ = False
        if args.use_non_normal_model:
            csvparser.remove_bn_to_csv(model_path, args.model)
    elif model_type == model.ModelType.MHATTENTION:
        LUTEN_ = True
        if args.use_non_normal_model:
            csvparser.remove_ln_to_csv(model_path, args.model)
    csvparser.add_pooling_to_csv(model_path, args.model, not args.use_non_normal_model)

    # get data
    layernum, model_nd = api.read_morax_csv(
        model_path, args.model, not args.use_non_normal_model
    )
    ModelDAG, model_list, concatlist = api.make_model(
        args.model, model_type, layernum, model_nd
    )
    api.add_layerclass_to_dag(ModelDAG, model_list, concatlist)

    # init morax obj
    MoraxChip = chip.MoraxChip()
    MemMonitor = memonitor.Memonitor()
    Mapper = mapper.Mapper(LUTEN_)
    Schduler = OL.Schduler()

    strategy = OFL.Strategy.greedy
    # run
    if args.scenario == "sm":
        # offline process
        CschduleList, OnRRAMLayerIndexList = Mapper.map_single(
            ModelDAG, MoraxChip, strategy, args.batch
        )
        Schduler.update_sch(CschduleList, OnRRAMLayerIndexList)
        query.generate_queries(ModelDAG, MoraxChip, args.batch)
        # online process
        MoraxChip.invoke_morax(ModelDAG, MemMonitor, Schduler)
    elif args.scenario == "mm":
        # TODO
        Mapper.map_multi(ModelDAG, MoraxChip)
    elif args.scenario == "cs":
        # TODO
        raise DataError

    # post process
    print("Here")
