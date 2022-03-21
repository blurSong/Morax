# Morax means the mixing of CMOS and RRAM
# Author: Tao Song.
# Last modification: 0316
# FrontEnd API

import os
import re
import sys
import numpy as np
import copy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import subprocess as SP
from torch._C import CONV_BN_FUSION
import morax.model.layer as Lyr
import morax.model.model as Model
from morax.frontend.csvparser import apd_is_index2
from morax.model.layer import MXLCD, MXLTD, MXNTD


def read_morax_csv(_modelpath, _modelname, _isbn=False):
    if _isbn is True:
        csv = _modelname + '_mora.csv'
    else:
        csv = _modelname + '.csv'
    morax_csv_path = os.path.abspath(os.path.join(_modelpath, csv))
    model_df = pd.read_csv(morax_csv_path)
    model_nd = model_df.to_numpy()
    layernum = model_nd.shape[0]
    return layernum, model_nd


def make_model(_model, _modeltype, _layernum, _model_nd):
    model_dag = Model.ModelDAG(_model, _modeltype)
    model_list = Model.LayerList(_model, _modeltype)
    for idx in range(_layernum):
        line = _model_nd[idx, ...]
        layertype = MXLTD[line[MXLCD['TYP']]] if line[MXLCD['TYP']] >= 0 else MXNTD[line[MXLCD['TYP']]]
        # add layer
        if layertype == 'Linear':
            layername = layertype + str(idx)
            linearlayer = Lyr.Linear(layername, idx, Lyr.LinearLayerType.Linear, line)
            model_list.add_layer(copy.deepcopy(linearlayer))
            model_dag.add_layer(idx, True)
        if layertype == 'CONV':
            layername = layertype + str(idx)
            linearlayer = Lyr.CONV(layername, idx, Lyr.LinearLayerType.CONV, line)
            model_list.add_layer(copy.deepcopy(linearlayer))
            model_dag.add_layer(idx, True)
        if layertype == 'DWCONV':
            layername = layertype + str(idx)
            linearlayer = Lyr.DWCONV(layername, idx, Lyr.LinearLayerType.DWCONV, line)
            model_list.add_layer(copy.deepcopy(linearlayer))
            model_dag.add_layer(idx, True)
        if layertype == 'Residual':
            layername = layertype + str(idx)
            linearlayer = Lyr.Residual(layername, idx, Lyr.LinearLayerType.Residual, line)
            model_list.add_layer(copy.deepcopy(linearlayer))
            model_dag.add_layer(idx, True)
        if layertype == 'Batchnorm':
            layername = layertype + str(idx)
            linearlayer = Lyr.Batchnorm(layername, idx, Lyr.LinearLayerType.Batchnorm, line)
            model_list.add_layer(copy.deepcopy(linearlayer))
            model_dag.add_layer(idx, True)
        if layertype == 'TRCONV':
            layername = layertype + str(idx)
            linearlayer = Lyr.TRCONV(layername, idx, Lyr.LinearLayerType.TRCONV, line)
            model_list.add_layer(copy.deepcopy(linearlayer))
            model_dag.add_layer(idx, True)
        if layertype == 'NGCONV':
            layername = layertype + str(idx)
            linearlayer = Lyr.NGCONV(layername, idx, Lyr.LinearLayerType.NGCONV, line)
            model_list.add_layer(copy.deepcopy(linearlayer))
            model_dag.add_layer(idx, True)
        if layertype == 'TRCONV':
            layername = layertype + str(idx)
            linearlayer = Lyr.TRCONV(layername, idx, Lyr.LinearLayerType.TRCONV, line)
            model_list.add_layer(copy.deepcopy(linearlayer))
            model_dag.add_layer(idx, True)
        if layertype == 'VDP':
            layername = layertype + str(idx)
            linearlayer = Lyr.VDP(layername, idx, Lyr.LinearLayerType.VDP, line)
            model_list.add_layer(copy.deepcopy(linearlayer))
            model_dag.add_layer(idx, True)
        if layertype == 'VADD':
            layername = layertype + str(idx)
            linearlayer = Lyr.VADD(layername, idx, Lyr.LinearLayerType.VADD, line)
            model_list.add_layer(copy.deepcopy(linearlayer))
            model_dag.add_layer(idx, True)
        if layertype == 'VMUL':
            layername = layertype + str(idx)
            linearlayer = Lyr.VMUL(layername, idx, Lyr.LinearLayerType.VMUL, line)
            model_list.add_layer(copy.deepcopy(linearlayer))
            model_dag.add_layer(idx, True)
        if layertype == 'GEMM':
            layername = layertype + str(idx)
            linearlayer = Lyr.GEMM(layername, idx, Lyr.LinearLayerType.GEMM, line)
            model_list.add_layer(copy.deepcopy(linearlayer))
            model_dag.add_layer(idx, True)
        if layertype == 'Layernorm':
            layername = layertype + str(idx)
            linearlayer = Lyr.Layernorm(layername, idx, Lyr.LinearLayerType.Layernorm, line)
            model_list.add_layer(copy.deepcopy(linearlayer))
            model_dag.add_layer(idx, True)
        if layertype == 'Pooling':
            layername = layertype + str(idx)
            nonlinearlayer = Lyr.Pooling(layername, idx, Lyr.NonlinearLayerType.Pooling, line)
            model_list.add_layer(copy.deepcopy(nonlinearlayer))
            model_dag.add_layer(idx, False)
        if layertype == 'Softmax1D':
            layername = layertype + str(idx)
            nonlinearlayer = Lyr.Softmax1D(layername, idx, Lyr.NonlinearLayerType.Softmax1D, line)
            model_list.add_layer(copy.deepcopy(nonlinearlayer))
            model_dag.add_layer(idx, False)
        if layertype == 'Softmax2D':
            layername = layertype + str(idx)
            nonlinearlayer = Lyr.Softmax2D(layername, idx, Lyr.NonlinearLayerType.Softmax2D, line)
            model_list.add_layer(copy.deepcopy(nonlinearlayer))
            model_dag.add_layer(idx, False)
        # add edge
        preidx = line[MXLCD['IDX']] + idx
        assert preidx < idx
        model_dag.add_edge(preidx, idx)
        if apd_is_index2(layertype, line[MXLCD['APD']]):
            apdidx = line[MXLCD['APD']] + idx
            assert apdidx < idx
            model_dag.add_edge(apdidx, idx)
    assert model_dag.layernum == model_list.layernum
    assert model_dag.layernum == _layernum
    return model_dag, model_list
