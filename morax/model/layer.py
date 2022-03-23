# Note:
# To simplify the VPU pooling strategy, Morax consider the activation opreator and pooling opreator as 2 characteristics of layer itself.
# We consider using model.csv instead of mode_mora.csv

import copy
import re
import sys
import os
from enum import Enum
import numpy as np
import pandas as pd
from pyrsistent import T
import torch
import torch.nn as nn
import torch.nn.functional as Func

morax_layer_csv_dicts = {
    "IC": 0,
    "OC": 1,
    "FS": 2,
    "KS": 3,
    "STR": 4,
    "TYP": 5,
    "RP": 6,
    "IDX": 7,
    "APD": 8,
}
# MXLCD
morax_linearlayer_type_dicts = {
    0: "Linear",
    1: "CONV",
    2: "DWCONV",
    3: "Residual",
    4: "Batchnorm",
    5: "TRCONV",
    6: "NGCONV",
    7: "VDP",
    8: "VADD",
    9: "VMUL",
    10: "GEMM",
    11: "Layernorm",
}  # MXLTD

morax_nonlinearlayer_type_dicts = {
    -1: "Pooling",
    -2: "Softmax1D",
    -3: "Softmax2D",
    # 1: 'relu',
    # 1: 'tanh',
    # 2: 'sigmoid',
}  # MXNTD

MXLCD = morax_layer_csv_dicts
MXLTD = morax_linearlayer_type_dicts
MXNTD = morax_nonlinearlayer_type_dicts


class ModelType(Enum):
    MLP = 0
    CNN = 1
    RNN = 2
    LSTM = 3
    MHATTENTION = 4


class LinearLayerType(Enum):
    Linear = 0
    CONV = 1
    DWCONV = 2
    Residual = 3
    Batchnorm = 4
    TRCONV = 5
    NGCONV = 6

    VDP = 7
    VADD = 8
    VMUL = 9
    GEMM = 10
    MADD = 11
    Layernorm = 12


class NonlinearLayerType(Enum):
    Pooling = -1
    Softmax1D = -2
    Softmax2D = -3


class Layer:
    def __init__(self, _layername, _layerindex) -> None:
        self.layer_name = _layername
        self.layer_index = _layerindex


class LinearLayer(Layer):
    def __init__(
        self, _layername, _layerindex, _layertype: LinearLayerType, _layercsvline
    ) -> None:
        Layer.__init__(self, _layername, _layerindex)
        self.layer_type = _layertype
        self.layer_csvline = _layercsvline


class Linear(LinearLayer):
    def __init__(self, _layername, _layerindex, _layertype, _layercsvline) -> None:
        LinearLayer.__init__(self, _layername, _layerindex, _layertype, _layercsvline)
        self.row_dim = self.layer_csvline[MXLCD["IC"]]
        self.col_dim = self.layer_csvline[MXLCD["OC"]]
        self.is_activated = False if self.layer_csvline[MXLCD["RP"]] == 0 else True


class CONV(LinearLayer):
    def __init__(self, _layername, _layerindex, _layertype, _layercsvline) -> None:
        LinearLayer.__init__(self, _layername, _layerindex, _layertype, _layercsvline)
        self.in_channel = self.layer_csvline[MXLCD["IC"]]
        self.out_channel = self.layer_csvline[MXLCD["OC"]]
        self.feature_size = self.layer_csvline[MXLCD["FS"]]
        self.kernel_size = self.layer_csvline[MXLCD["KS"]]
        self.stride = self.layer_csvline[MXLCD["STR"]]
        self.is_activated = False if self.layer_csvline[MXLCD["RP"]] == 0 else True
        self.input_indecies_tuple = (
            self.layer_csvline[MXLCD["IDX"]],
            self.layer_csvline[MXLCD["APD"]],
        )


class DWCONV(LinearLayer):
    def __init__(self, _layername, _layerindex, _layertype, _layercsvline) -> None:
        LinearLayer.__init__(self, _layername, _layerindex, _layertype, _layercsvline)
        self.channel = self.layer_csvline[MXLCD["IC"]]
        self.feature_size = self.layer_csvline[MXLCD["FS"]]
        self.kernel_size = self.layer_csvline[MXLCD["KS"]]
        self.stride = self.layer_csvline[MXLCD["STR"]]
        self.is_activated = False if self.layer_csvline[MXLCD["RP"]] == 0 else True
        self.input_indecies_tuple = (self.layer_csvline[MXLCD["IDX"]], 0)


class Residual(LinearLayer):
    def __init__(self, _layername, _layerindex, _layertype, _layercsvline) -> None:
        LinearLayer.__init__(self, _layername, _layerindex, _layertype, _layercsvline)
        self.channel = self.layer_csvline[MXLCD["IC"]]
        self.feature_size = self.layer_csvline[MXLCD["FS"]]
        self.is_activated = False if self.layer_csvline[MXLCD["RP"]] == 0 else True
        self.input_indecies_tuple = (
            self.layer_csvline[MXLCD["IDX"]],
            self.layer_csvline[MXLCD["APD"]],
        )


class Batchnorm(LinearLayer):
    def __init__(self, _layername, _layerindex, _layertype, _layercsvline) -> None:
        LinearLayer.__init__(self, _layername, _layerindex, _layertype, _layercsvline)
        self.channel = self.layer_csvline[MXLCD["IC"]]
        self.feature_size = self.layer_csvline[MXLCD["FS"]]
        self.is_activated = False if self.layer_csvline[MXLCD["RP"]] == 0 else True
        self.input_indecies_tuple = (self.layer_csvline[MXLCD["IDX"]], 0)


class TRCONV(LinearLayer):
    def __init__(self, _layername, _layerindex, _layertype, _layercsvline) -> None:
        LinearLayer.__init__(self, _layername, _layerindex, _layertype, _layercsvline)
        self.in_channel = self.layer_csvline[MXLCD["IC"]]
        self.out_channel = self.layer_csvline[MXLCD["OC"]]
        self.feature_size = self.layer_csvline[MXLCD["FS"]]
        self.kernel_size = self.layer_csvline[MXLCD["KS"]]
        self.stride = self.layer_csvline[MXLCD["STR"]]
        self.dilation = self.layer_csvline[MXLCD["APD"]]
        self.is_activated = False if self.layer_csvline[MXLCD["RP"]] == 0 else True
        self.input_indecies_tuple = (self.layer_csvline[MXLCD["IDX"]], 0)


class NGCONV(LinearLayer):
    def __init__(self, _layername, _layerindex, _layertype, _layercsvline) -> None:
        LinearLayer.__init__(self, _layername, _layerindex, _layertype, _layercsvline)
        self.in_channel = self.layer_csvline[MXLCD["IC"]]
        self.out_channel = self.layer_csvline[MXLCD["OC"]]
        self.feature_size = self.layer_csvline[MXLCD["FS"]]
        self.kernel_size = self.layer_csvline[MXLCD["KS"]]
        self.stride = self.layer_csvline[MXLCD["STR"]]
        self.group = self.layer_csvline[MXLCD["APD"]]
        self.is_activated = False if self.layer_csvline[MXLCD["RP"]] == 0 else True
        self.input_indecies_tuple = (self.layer_csvline[MXLCD["IDX"]], 0)


class VDP(LinearLayer):
    def __init__(self, _layername, _layerindex, _layertype, _layercsvline) -> None:
        LinearLayer.__init__(self, _layername, _layerindex, _layertype, _layercsvline)
        self.v_dim = self.layer_csvline[MXLCD["IC"]]
        self.is_activated = False if self.layer_csvline[MXLCD["RP"]] == 0 else True
        self.input_indecies_tuple = (
            self.layer_csvline[MXLCD["IDX"]],
            self.layer_csvline[MXLCD["APD"]],
        )


class VADD(LinearLayer):
    def __init__(self, _layername, _layerindex, _layertype, _layercsvline) -> None:
        LinearLayer.__init__(self, _layername, _layerindex, _layertype, _layercsvline)
        self.v_dim = self.layer_csvline[MXLCD["IC"]]
        self.is_activated = False if self.layer_csvline[MXLCD["RP"]] == 0 else True
        self.input_indecies_tuple = (
            self.layer_csvline[MXLCD["IDX"]],
            self.layer_csvline[MXLCD["APD"]],
        )


class VMUL(LinearLayer):
    def __init__(self, _layername, _layerindex, _layertype, _layercsvline) -> None:
        LinearLayer.__init__(self, _layername, _layerindex, _layertype, _layercsvline)
        self.v_dim = self.layer_csvline[MXLCD["IC"]]
        self.is_activated = False if self.layer_csvline[MXLCD["RP"]] == 0 else True
        self.input_indecies_tuple = (
            self.layer_csvline[MXLCD["IDX"]],
            self.layer_csvline[MXLCD["APD"]],
        )


class GEMM(LinearLayer):
    def __init__(self, _layername, _layerindex, _layertype, _layercsvline) -> None:
        LinearLayer.__init__(self, _layername, _layerindex, _layertype, _layercsvline)
        # MK * KN
        self.m_dim = self.layer_csvline[MXLCD["IC"]]
        self.n_dim = self.layer_csvline[MXLCD["OC"]]
        self.k_dim = self.layer_csvline[MXLCD["FS"]]
        self.is_activated = False if self.layer_csvline[MXLCD["RP"]] == 0 else True
        self.input_indecies_tuple = (
            self.layer_csvline[MXLCD["IDX"]],
            self.layer_csvline[MXLCD["APD"]],
        )


class MADD(LinearLayer):
    def __init__(self, _layername, _layerindex, _layertype, _layercsvline) -> None:
        LinearLayer.__init__(self, _layername, _layerindex, _layertype, _layercsvline)
        # MK * KN
        self.m_dim = self.layer_csvline[MXLCD["IC"]]
        self.n_dim = self.layer_csvline[MXLCD["OC"]]
        self.k_dim = self.layer_csvline[MXLCD["FS"]]
        self.is_activated = False if self.layer_csvline[MXLCD["RP"]] == 0 else True
        self.input_indecies_tuple = (
            self.layer_csvline[MXLCD["IDX"]],
            self.layer_csvline[MXLCD["APD"]],
        )


class Layernorm(LinearLayer):
    def __init__(self, _layername, _layerindex, _layertype, _layercsvline) -> None:
        LinearLayer.__init__(self, _layername, _layerindex, _layertype, _layercsvline)
        self.channel = self.layer_csvline[MXLCD["IC"]]
        self.feature_size = self.layer_csvline[MXLCD["FS"]]
        self.is_activated = False if self.layer_csvline[MXLCD["RP"]] == 0 else True
        self.input_indecies_tuple = (self.layer_csvline[MXLCD["IDX"]], 0)


class NonlinearLayer(Layer):
    def __init__(
        self, _layername, _layerindex, _layertype: NonlinearLayerType, _layercsvline
    ) -> None:
        Layer.__init__(self, _layername, _layerindex)
        self.layer_type = _layertype
        self.layer_csvline = _layercsvline


class Pooling(NonlinearLayer):
    def __init__(self, _layername, _layerindex, _layertype, _layercsvline) -> None:
        NonlinearLayer.__init__(
            self, _layername, _layerindex, _layertype, _layercsvline
        )
        self.channel = self.layer_csvline[MXLCD["IC"]]
        self.feature_size = self.layer_csvline[MXLCD["FS"]]
        self.kernel_size = self.layer_csvline[MXLCD["KS"]]
        self.stride = self.layer_csvline[MXLCD["STR"]]
        self.is_activated = False if self.layer_csvline[MXLCD["RP"]] == 0 else True
        self.input_indecies_tuple = (self.layer_csvline[MXLCD["IDX"]], 0)


class Softmax1D(NonlinearLayer):
    def __init__(self, _layername, _layerindex, _layertype, _layercsvline) -> None:
        NonlinearLayer.__init__(
            self, _layername, _layerindex, _layertype, _layercsvline
        )
        self.v_dim = self.layer_csvline[MXLCD["IC"]]
        self.is_activated = False if self.layer_csvline[MXLCD["RP"]] == 0 else True
        self.input_indecies_tuple = (self.layer_csvline[MXLCD["IDX"]], 0)


class Softmax2D(NonlinearLayer):
    def __init__(self, _layername, _layerindex, _layertype, _layercsvline) -> None:
        NonlinearLayer.__init__(
            self, _layername, _layerindex, _layertype, _layercsvline
        )
        self.row_dim = self.layer_csvline[MXLCD["IC"]]
        self.col_dim = self.layer_csvline[MXLCD["OC"]]
        self.is_activated = False if self.layer_csvline[MXLCD["RP"]] == 0 else True
        self.input_indecies_tuple = (self.layer_csvline[MXLCD["IDX"]], 0)

