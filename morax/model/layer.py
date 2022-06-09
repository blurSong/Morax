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

morax_layer_csv_dicts_cnn = {
    "IC": 0,
    "OC": 1,
    "FS": 2,
    "KS": 3,
    "STR": 4,
    "TYP": 5,
    "RP": 6,
    "IDX": 7,
    "APD": 8,
}  # mxLCD_CNN

morax_gemm_layer_csv_dicts_gemm = {
    "M": 0,
    "N": 1,
    "K": 2,
    "TYP": 3,
    "ACT": 4,
    "IDX1": 5,
    "IDX2": 6,
}  # mxLCD_GEMM

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
    10: "VMM",
    11: "GEMM",
    12: "MADD",
    13: "Layernorm",
    14: "CONCAT",
}  # mxLTD

morax_nonlinearlayer_type_dicts = {
    -1: "Pooling",
    -2: "Softmax1D",
    -3: "Softmax2D",
    # 1: 'relu',
    # 1: 'tanh',
    # 2: 'sigmoid',
}  # mxNTD

mxLCD_CNN = morax_layer_csv_dicts_cnn
mxLCD_GEMM = morax_gemm_layer_csv_dicts_gemm
mxLTD = morax_linearlayer_type_dicts
mxNTD = morax_nonlinearlayer_type_dicts


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
    VMM = 10
    GEMM = 11
    MADD = 12
    Layernorm = 13
    CONCAT = 14


class NonlinearLayerType(Enum):
    Pooling = -1
    Softmax1D = -2
    Softmax2D = -3


class Layer:
    def __init__(self, _layername: str, _layerindex) -> None:
        self.layer_name = _layername
        self.layer_index = _layerindex

    def change_layerinfo(self, _newindex):
        end = 0
        for i in range(len(self.layer_name) - 1, 0):
            if self.layer_name[i].isdigit():
                continue
            else:
                end = i
                break
        self.layer_name = self.layer_name[: end + 1] + str(_newindex)
        self.layer_index = _newindex


class LinearLayer(Layer):
    def __init__(
        self, _layername, _layerindex, _layertype: LinearLayerType, _layercsvline
    ) -> None:
        Layer.__init__(self, _layername, _layerindex)
        self.layer_type = _layertype
        self.layer_csvline = _layercsvline


"""CONV based Models"""


class Linear(LinearLayer):
    def __init__(self, _layername, _layerindex, _layertype, _layercsvline) -> None:
        LinearLayer.__init__(self, _layername, _layerindex, _layertype, _layercsvline)
        self.row_dim = self.layer_csvline[mxLCD_CNN["IC"]]
        self.col_dim = self.layer_csvline[mxLCD_CNN["OC"]]
        self.is_activated = False if self.layer_csvline[mxLCD_CNN["RP"]] == 0 else True
        self.is_fc = False if self.layer_csvline[mxLCD_CNN["APD"]] == 0 else True
        self.input_indecies_tuple = (self.layer_csvline[mxLCD_CNN["IDX"]], 0)


class CONV(LinearLayer):
    def __init__(self, _layername, _layerindex, _layertype, _layercsvline) -> None:
        LinearLayer.__init__(self, _layername, _layerindex, _layertype, _layercsvline)
        self.in_channel = self.layer_csvline[mxLCD_CNN["IC"]]
        self.out_channel = self.layer_csvline[mxLCD_CNN["OC"]]
        self.feature_size = self.layer_csvline[mxLCD_CNN["FS"]]
        self.kernel_size = self.layer_csvline[mxLCD_CNN["KS"]]
        self.stride = self.layer_csvline[mxLCD_CNN["STR"]]
        self.is_activated = False if self.layer_csvline[mxLCD_CNN["RP"]] == 0 else True
        self.input_indecies_tuple = (
            self.layer_csvline[mxLCD_CNN["IDX"]],
            self.layer_csvline[mxLCD_CNN["APD"]],
        )


class DWCONV(LinearLayer):
    def __init__(self, _layername, _layerindex, _layertype, _layercsvline) -> None:
        LinearLayer.__init__(self, _layername, _layerindex, _layertype, _layercsvline)
        self.channel = self.layer_csvline[mxLCD_CNN["IC"]]
        self.feature_size = self.layer_csvline[mxLCD_CNN["FS"]]
        self.kernel_size = self.layer_csvline[mxLCD_CNN["KS"]]
        self.stride = self.layer_csvline[mxLCD_CNN["STR"]]
        self.is_activated = False if self.layer_csvline[mxLCD_CNN["RP"]] == 0 else True
        self.input_indecies_tuple = (self.layer_csvline[mxLCD_CNN["IDX"]], 0)


class Residual(LinearLayer):
    def __init__(self, _layername, _layerindex, _layertype, _layercsvline) -> None:
        LinearLayer.__init__(self, _layername, _layerindex, _layertype, _layercsvline)
        self.channel = self.layer_csvline[mxLCD_CNN["IC"]]
        self.feature_size = self.layer_csvline[mxLCD_CNN["FS"]]
        self.is_activated = False if self.layer_csvline[mxLCD_CNN["RP"]] == 0 else True
        self.input_indecies_tuple = (
            self.layer_csvline[mxLCD_CNN["IDX"]],
            self.layer_csvline[mxLCD_CNN["APD"]],
        )


class Batchnorm(LinearLayer):
    def __init__(self, _layername, _layerindex, _layertype, _layercsvline) -> None:
        LinearLayer.__init__(self, _layername, _layerindex, _layertype, _layercsvline)
        self.channel = self.layer_csvline[mxLCD_CNN["IC"]]
        self.feature_size = self.layer_csvline[mxLCD_CNN["FS"]]
        self.is_activated = False if self.layer_csvline[mxLCD_CNN["RP"]] == 0 else True
        self.input_indecies_tuple = (self.layer_csvline[mxLCD_CNN["IDX"]], 0)


class TRCONV(LinearLayer):
    def __init__(self, _layername, _layerindex, _layertype, _layercsvline) -> None:
        LinearLayer.__init__(self, _layername, _layerindex, _layertype, _layercsvline)
        self.in_channel = self.layer_csvline[mxLCD_CNN["IC"]]
        self.out_channel = self.layer_csvline[mxLCD_CNN["OC"]]
        self.feature_size = self.layer_csvline[mxLCD_CNN["FS"]]
        self.kernel_size = self.layer_csvline[mxLCD_CNN["KS"]]
        self.stride = self.layer_csvline[mxLCD_CNN["STR"]]
        self.dilation = self.layer_csvline[mxLCD_CNN["APD"]]
        self.is_activated = False if self.layer_csvline[mxLCD_CNN["RP"]] == 0 else True
        self.input_indecies_tuple = (self.layer_csvline[mxLCD_CNN["IDX"]], 0)


class NGCONV(LinearLayer):
    def __init__(self, _layername, _layerindex, _layertype, _layercsvline) -> None:
        LinearLayer.__init__(self, _layername, _layerindex, _layertype, _layercsvline)
        self.in_channel = self.layer_csvline[mxLCD_CNN["IC"]]
        self.out_channel = self.layer_csvline[mxLCD_CNN["OC"]]
        self.feature_size = self.layer_csvline[mxLCD_CNN["FS"]]
        self.kernel_size = self.layer_csvline[mxLCD_CNN["KS"]]
        self.stride = self.layer_csvline[mxLCD_CNN["STR"]]
        self.group = self.layer_csvline[mxLCD_CNN["APD"]]
        self.is_activated = False if self.layer_csvline[mxLCD_CNN["RP"]] == 0 else True
        self.input_indecies_tuple = (self.layer_csvline[mxLCD_CNN["IDX"]], 0)


"""GEMM based Models"""


class VDP(LinearLayer):
    def __init__(self, _layername, _layerindex, _layertype, _layercsvline) -> None:
        LinearLayer.__init__(self, _layername, _layerindex, _layertype, _layercsvline)
        self.v_dim = self.layer_csvline[mxLCD_GEMM["M"]]
        self.is_activated = (
            False if self.layer_csvline[mxLCD_GEMM["ACT"]] == 0 else True
        )
        self.input_indecies_tuple = (
            self.layer_csvline[mxLCD_GEMM["IDX1"]],
            self.layer_csvline[mxLCD_GEMM["IDX2"]],
        )


class VADD(LinearLayer):
    def __init__(self, _layername, _layerindex, _layertype, _layercsvline) -> None:
        LinearLayer.__init__(self, _layername, _layerindex, _layertype, _layercsvline)
        self.v_dim = self.layer_csvline[mxLCD_GEMM["M"]]
        self.is_activated = (
            False if self.layer_csvline[mxLCD_GEMM["ACT"]] == 0 else True
        )
        self.input_indecies_tuple = (
            self.layer_csvline[mxLCD_GEMM["IDX1"]],
            self.layer_csvline[mxLCD_GEMM["IDX2"]],
        )


class VMUL(LinearLayer):
    def __init__(self, _layername, _layerindex, _layertype, _layercsvline) -> None:
        LinearLayer.__init__(self, _layername, _layerindex, _layertype, _layercsvline)
        self.v_dim = self.layer_csvline[mxLCD_GEMM["M"]]
        self.is_activated = (
            False if self.layer_csvline[mxLCD_GEMM["ACT"]] == 0 else True
        )
        self.input_indecies_tuple = (
            self.layer_csvline[mxLCD_GEMM["IDX1"]],
            self.layer_csvline[mxLCD_GEMM["IDX2"]],
        )


class VMM(LinearLayer):
    def __init__(self, _layername, _layerindex, _layertype, _layercsvline) -> None:
        LinearLayer.__init__(self, _layername, _layerindex, _layertype, _layercsvline)
        self.v_dim = self.layer_csvline[mxLCD_GEMM["M"]]
        self.row_dim = self.v_dim
        self.col_dim = self.layer_csvline[mxLCD_GEMM["N"]]
        self.is_activated = (
            False if self.layer_csvline[mxLCD_GEMM["ACT"]] == 0 else True
        )
        self.input_indecies_tuple = (
            self.layer_csvline[mxLCD_GEMM["IDX1"]],
            self.layer_csvline[mxLCD_GEMM["IDX2"]],
        )


class GEMM(LinearLayer):
    def __init__(self, _layername, _layerindex, _layertype, _layercsvline) -> None:
        LinearLayer.__init__(self, _layername, _layerindex, _layertype, _layercsvline)
        # MK * KN
        self.m_dim = self.layer_csvline[mxLCD_GEMM["M"]]
        self.n_dim = self.layer_csvline[mxLCD_GEMM["N"]]
        self.k_dim = self.layer_csvline[mxLCD_GEMM["K"]]
        self.is_activated = (
            False if self.layer_csvline[mxLCD_GEMM["ACT"]] == 0 else True
        )
        self.input_indecies_tuple = (
            self.layer_csvline[mxLCD_GEMM["IDX1"]],
            self.layer_csvline[mxLCD_GEMM["IDX2"]],
        )


class MADD(LinearLayer):
    def __init__(self, _layername, _layerindex, _layertype, _layercsvline) -> None:
        LinearLayer.__init__(self, _layername, _layerindex, _layertype, _layercsvline)
        # MK * KN
        self.row_dim = self.layer_csvline[mxLCD_GEMM["M"]]
        self.col_dim = self.layer_csvline[mxLCD_GEMM["N"]]
        self.is_activated = (
            False if self.layer_csvline[mxLCD_GEMM["ACT"]] == 0 else True
        )
        self.input_indecies_tuple = (
            self.layer_csvline[mxLCD_GEMM["IDX1"]],
            self.layer_csvline[mxLCD_GEMM["IDX2"]],
        )


class Layernorm(LinearLayer):
    def __init__(self, _layername, _layerindex, _layertype, _layercsvline) -> None:
        LinearLayer.__init__(self, _layername, _layerindex, _layertype, _layercsvline)
        self.row_dim = self.layer_csvline[mxLCD_GEMM["M"]]
        self.col_dim = self.layer_csvline[mxLCD_GEMM["N"]]
        self.is_activated = (
            False if self.layer_csvline[mxLCD_GEMM["ACT"]] == 0 else True
        )
        self.input_indecies_tuple = (self.layer_csvline[mxLCD_GEMM["IDX1"]], 0)


class CONCAT(LinearLayer):
    def __init__(self, _layername, _layerindex, _layertype, _layercsvline) -> None:
        LinearLayer.__init__(self, _layername, _layerindex, _layertype, _layercsvline)
        self.head = self.layer_csvline[mxLCD_GEMM["M"]]
        self.input_indecies_tuple = (self.layer_csvline[mxLCD_GEMM["IDX1"]], 0)


"""NON-LINEAR LAYER TYPE"""


class NonlinearLayer(Layer):
    def __init__(
        self, _layername, _layerindex, _layertype: NonlinearLayerType, _layercsvline
    ) -> None:
        Layer.__init__(self, _layername, _layerindex)
        self.layer_type = _layertype
        self.layer_csvline = _layercsvline


class Pooling(NonlinearLayer):  # CNN
    def __init__(self, _layername, _layerindex, _layertype, _layercsvline) -> None:
        NonlinearLayer.__init__(
            self, _layername, _layerindex, _layertype, _layercsvline
        )
        self.channel = self.layer_csvline[mxLCD_CNN["IC"]]
        self.feature_size = self.layer_csvline[mxLCD_CNN["FS"]]
        self.kernel_size = self.layer_csvline[mxLCD_CNN["KS"]]
        self.stride = self.layer_csvline[mxLCD_CNN["STR"]]
        self.is_activated = False if self.layer_csvline[mxLCD_CNN["RP"]] == 0 else True
        self.input_indecies_tuple = (self.layer_csvline[mxLCD_CNN["IDX"]], 0)


class Softmax1D(NonlinearLayer):  # MLP
    def __init__(self, _layername, _layerindex, _layertype, _layercsvline) -> None:
        NonlinearLayer.__init__(
            self, _layername, _layerindex, _layertype, _layercsvline
        )
        self.v_dim = self.layer_csvline[mxLCD_GEMM["M"]]
        self.is_activated = (
            False if self.layer_csvline[mxLCD_GEMM["ACT"]] == 0 else True
        )
        self.input_indecies_tuple = (self.layer_csvline[mxLCD_GEMM["IDX1"]], 0)


class Softmax2D(NonlinearLayer):  # MLP
    def __init__(self, _layername, _layerindex, _layertype, _layercsvline) -> None:
        NonlinearLayer.__init__(
            self, _layername, _layerindex, _layertype, _layercsvline
        )
        self.row_dim = self.layer_csvline[mxLCD_GEMM["M"]]
        self.col_dim = self.layer_csvline[mxLCD_GEMM["N"]]
        self.is_activated = (
            False if self.layer_csvline[mxLCD_GEMM["ACT"]] == 0 else True
        )
        self.input_indecies_tuple = (self.layer_csvline[mxLCD_GEMM["IDX1"]], 0)

