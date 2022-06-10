# Morax means the mixing of CMOS and RRAM
# Author: Tao Song.
# Last modification: 0316
# FrontEnd API

import os
import random
import copy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import subprocess as SP
from sqlite3 import DataError


import morax.model.layer as Lyr
import morax.model.model as Model
from morax.model.layer import (
    LinearLayerType as LLT,
    NonlinearLayerType as NLT,
    mxLCD_CNN,
    mxLCD_GEMM,
    mxLTD,
    mxNTD,
)
from morax.system.config import MoraxConfig


def get_layer_scratchdict(_layerclass):
    layertype = _layerclass.layer_type
    if layertype in [LLT.Linear, LLT.VMM]:
        scratchpad_dict = {"M": _layerclass.col_dim}
    elif layertype in [LLT.CONV, LLT.NGCONV]:
        scratchpad_dict = {
            "C": _layerclass.out_channel,
            "H": _layerclass.feature_size / _layerclass.stride,
            "W": _layerclass.feature_size / _layerclass.stride,
        }
    elif layertype == LLT.TRCONV:
        omapsize = (
            _layerclass.feature_size - 1
        ) * _layerclass.stride + _layerclass.kernel_size
        scratchpad_dict = {
            "C": _layerclass.out_channel,
            "H": omapsize,
            "W": omapsize,
        }
    elif layertype == LLT.DWCONV:
        scratchpad_dict = {
            "C": _layerclass.channel,
            "H": _layerclass.feature_size / _layerclass.stride,
            "W": _layerclass.feature_size / _layerclass.stride,
        }
    elif layertype in [LLT.Residual, LLT.Batchnorm]:
        scratchpad_dict = {
            "C": _layerclass.channel,
            "H": _layerclass.feature_size,
            "W": _layerclass.feature_size,
        }
    elif layertype in [LLT.VDP, LLT.VADD]:
        scratchpad_dict = {"M": _layerclass.v_dim}
    elif layertype == LLT.GEMM:
        scratchpad_dict = {"M": _layerclass.m_dim, "N": _layerclass.n_dim}
    elif layertype in [LLT.MADD, LLT.Layernorm]:
        scratchpad_dict = {"M": _layerclass.row_dim, "N": _layerclass.col_dim}
    # elif layertype == LLT.CONCAT:
    #    scratchpad_dict = {}
    elif layertype == NLT.Pooling:
        scratchpad_dict = {
            "C": _layerclass.channel,
            "H": _layerclass.feature_size / _layerclass.stride,
            "W": _layerclass.feature_size / _layerclass.stride,
        }
    elif layertype == NLT.Softmax1D:
        scratchpad_dict = {"M": _layerclass.v_dim}
    elif layertype == NLT.Softmax2D:
        scratchpad_dict = {"M": _layerclass.row_dim, "N": _layerclass.col_dim}
    return scratchpad_dict


def get_datatype(_layerclass):
    layertype = _layerclass.layer_type
    if layertype in [LLT.Linear, LLT.VDP, LLT.VADD, LLT.VMUL, LLT.VMM]:
        datatype = "VEC"
    elif layertype in [
        LLT.CONV,
        LLT.DWCONV,
        LLT.Residual,
        LLT.Batchnorm,
        LLT.TRCONV,
        LLT.NGCONV,
    ]:
        datatype = "FTR"
    elif layertype in [LLT.GEMM, LLT.MADD, LLT.Layernorm]:
        datatype = "MAT"
    elif layertype == NLT.Pooling:
        datatype = "FTR"
    elif layertype == NLT.Softmax2D:
        datatype = "MAT"
    elif layertype == NLT.Softmax1D:
        datatype = "VEC"
    return datatype


def get_weight_datatype(_layerclass):
    layertype = _layerclass.layer_type
    IIDX_L = _layerclass.input_indecies_tuple[0]
    IIDX_R = _layerclass.input_indecies_tuple[1]
    if layertype in [LLT.VDP, LLT.VADD, LLT.VMUL]:
        datatype = "VEC"
    elif layertype in [
        LLT.CONV,
        LLT.DWCONV,
        LLT.TRCONV,
        LLT.NGCONV,
    ]:
        datatype = "WET"
    elif layertype in [LLT.GEMM, LLT.MADD, LLT.VMM, LLT.Linear]:
        datatype = "MAT"
    return datatype


def get_weight_scratchdict(_layerclass):
    layertype = _layerclass.layer_type
    if layertype in [LLT.Linear, LLT.VMM]:
        scratchpad_dict = {"M": _layerclass.row_dim, "N": _layerclass.col_dim}
    elif layertype in [LLT.CONV, LLT.NGCONV, LLT.TRCONV]:
        scratchpad_dict = {
            "K": _layerclass.in_channel,
            "C": _layerclass.out_channel,
            "R": _layerclass.kernel_size,
            "S": _layerclass.feature_size,
        }
    elif layertype == LLT.DWCONV:
        scratchpad_dict = {
            "K": _layerclass.channel,
            "C": _layerclass.channel,
            "R": _layerclass.kernel_size,
            "S": _layerclass.feature_size,
        }
    elif layertype in [LLT.VDP, LLT.VADD]:
        scratchpad_dict = {"M": _layerclass.v_dim}
    elif layertype == LLT.GEMM:
        if _layerclass.input_indecies_tuple[1] == 0:
            scratchpad_dict = {"M": _layerclass.k_dim, "N": _layerclass.n_dim}
        else:
            scratchpad_dict = {"M": _layerclass.m_dim, "N": _layerclass.k_dim}
    elif layertype == LLT.MADD:
        scratchpad_dict = {"M": _layerclass.row_dim, "N": _layerclass.col_dim}
    return scratchpad_dict


def get_lookup_adress(_index, _chn):
    # NOTE fake func because no real data
    # _nvtcid: int, _clusterid: int, _sliceidlist: Anyc
    random.seed(_index)
    nvtcid = (
        random.randint(0, MoraxConfig.NVTCNum - 1) + _chn % MoraxConfig.NVTCNum
    ) % MoraxConfig.NVTCNum
    clusterid = (
        random.randint(0, MoraxConfig.ClusterNum - 1) + _chn % MoraxConfig.ClusterNum
    ) % MoraxConfig.ClusterNum
    sliceidlist = [
        (
            random.randint(0, MoraxConfig.RRAMSliceNum - 1)
            + _chn % MoraxConfig.RRAMSliceNum
        )
        % MoraxConfig.RRAMSliceNum
    ]
    return (nvtcid, clusterid, sliceidlist)


def read_morax_csv(_modelpath, _modelname, _isnorm=False):
    if _isnorm is True:
        csv = _modelname + "_norm.csv"
    else:
        csv = _modelname + ".csv"
    morax_csv_path = os.path.abspath(os.path.join(_modelpath, csv))
    model_df = pd.read_csv(morax_csv_path)
    model_nd = model_df.to_numpy()
    layernum = model_nd.shape[0]
    return layernum, model_nd


def make_model(_model, _modeltype, _layernum, _model_nd):
    model_dag = Model.ModelDAG(_model, _modeltype)
    model_list = Model.ModelList(_model, _modeltype)
    model_dag.add_vlayer()
    # model_list.add_vlayer()

    if _modeltype == Model.ModelType.MHATTENTION:
        outofrange_idx = _layernum
        concatlist = []

    for idx in range(_layernum):
        line = _model_nd[idx, ...]
        typeint = (
            mxLCD_CNN["TYP"] if _modeltype == Model.ModelType.CNN else mxLCD_GEMM["TYP"]
        )
        layertype = mxLTD[line[typeint]] if line[typeint] >= 0 else mxNTD[line[typeint]]
        # add layer
        if layertype == "Linear":
            layername = layertype + str(idx)
            linearlayer = Lyr.Linear(layername, idx, Lyr.LinearLayerType.Linear, line)
            model_list.add_layer(copy.deepcopy(linearlayer))
            model_dag.add_layer(idx, True)
        if layertype == "CONV":
            layername = layertype + str(idx)
            linearlayer = Lyr.CONV(layername, idx, Lyr.LinearLayerType.CONV, line)
            model_list.add_layer(copy.deepcopy(linearlayer))
            model_dag.add_layer(idx, True)
        if layertype == "DWCONV":
            layername = layertype + str(idx)
            linearlayer = Lyr.DWCONV(layername, idx, Lyr.LinearLayerType.DWCONV, line)
            model_list.add_layer(copy.deepcopy(linearlayer))
            model_dag.add_layer(idx, True)
        if layertype == "Residual":
            layername = layertype + str(idx)
            linearlayer = Lyr.Residual(
                layername, idx, Lyr.LinearLayerType.Residual, line
            )
            model_list.add_layer(copy.deepcopy(linearlayer))
            model_dag.add_layer(idx, True)
        if layertype == "Batchnorm":
            layername = layertype + str(idx)
            linearlayer = Lyr.Batchnorm(
                layername, idx, Lyr.LinearLayerType.Batchnorm, line
            )
            model_list.add_layer(copy.deepcopy(linearlayer))
            model_dag.add_layer(idx, True)
        if layertype == "TRCONV":
            layername = layertype + str(idx)
            linearlayer = Lyr.TRCONV(layername, idx, Lyr.LinearLayerType.TRCONV, line)
            model_list.add_layer(copy.deepcopy(linearlayer))
            model_dag.add_layer(idx, True)
        if layertype == "NGCONV":
            layername = layertype + str(idx)
            linearlayer = Lyr.NGCONV(layername, idx, Lyr.LinearLayerType.NGCONV, line)
            model_list.add_layer(copy.deepcopy(linearlayer))
            model_dag.add_layer(idx, True)
        if layertype == "TRCONV":
            layername = layertype + str(idx)
            linearlayer = Lyr.TRCONV(layername, idx, Lyr.LinearLayerType.TRCONV, line)
            model_list.add_layer(copy.deepcopy(linearlayer))
            model_dag.add_layer(idx, True)
        if layertype == "VDP":
            layername = layertype + str(idx)
            linearlayer = Lyr.VDP(layername, idx, Lyr.LinearLayerType.VDP, line)
            model_list.add_layer(copy.deepcopy(linearlayer))
            model_dag.add_layer(idx, True)
        if layertype == "VADD":
            layername = layertype + str(idx)
            linearlayer = Lyr.VADD(layername, idx, Lyr.LinearLayerType.VADD, line)
            model_list.add_layer(copy.deepcopy(linearlayer))
            model_dag.add_layer(idx, True)
        if layertype == "VMUL":
            layername = layertype + str(idx)
            linearlayer = Lyr.VMUL(layername, idx, Lyr.LinearLayerType.VMUL, line)
            model_list.add_layer(copy.deepcopy(linearlayer))
            model_dag.add_layer(idx, True)
        if layertype == "VMM":
            layername = layertype + str(idx)
            linearlayer = Lyr.VMM(layername, idx, Lyr.LinearLayerType.VMM, line)
            model_list.add_layer(copy.deepcopy(linearlayer))
            model_dag.add_layer(idx, True)
        if layertype == "GEMM":
            layername = layertype + str(idx)
            linearlayer = Lyr.GEMM(layername, idx, Lyr.LinearLayerType.GEMM, line)
            model_list.add_layer(copy.deepcopy(linearlayer))
            model_dag.add_layer(idx, True)
        if layertype == "MADD":
            layername = layertype + str(idx)
            linearlayer = Lyr.MADD(layername, idx, Lyr.LinearLayerType.MADD, line)
            model_list.add_layer(copy.deepcopy(linearlayer))
            model_dag.add_layer(idx, True)
        if layertype == "Layernorm":
            layername = layertype + str(idx)
            linearlayer = Lyr.Layernorm(
                layername, idx, Lyr.LinearLayerType.Layernorm, line
            )
            model_list.add_layer(copy.deepcopy(linearlayer))
            model_dag.add_layer(idx, True)
        if layertype == "Pooling":
            layername = layertype + str(idx)
            nonlinearlayer = Lyr.Pooling(
                layername, idx, Lyr.NonlinearLayerType.Pooling, line
            )
            model_list.add_layer(copy.deepcopy(nonlinearlayer))
            model_dag.add_layer(idx, False)
        if layertype == "Softmax1D":
            layername = layertype + str(idx)
            nonlinearlayer = Lyr.Softmax1D(
                layername, idx, Lyr.NonlinearLayerType.Softmax1D, line
            )
            model_list.add_layer(copy.deepcopy(nonlinearlayer))
            model_dag.add_layer(idx, False)
        if layertype == "Softmax2D":
            layername = layertype + str(idx)
            nonlinearlayer = Lyr.Softmax2D(
                layername, idx, Lyr.NonlinearLayerType.Softmax2D, line
            )
            model_list.add_layer(copy.deepcopy(nonlinearlayer))
            model_dag.add_layer(idx, False)
        if layertype == "CONCAT":
            # KEYPOINT: INSERT MULTI-HEAD add vIDX
            layername = layertype + str(idx)
            linearlayer = Lyr.CONCAT(
                layername, idx, Lyr.LinearLayerType.Layernorm, line
            )
            model_list.add_layer(copy.deepcopy(linearlayer))
            model_dag.add_layer(idx, True)
            #
            # NOW add other heads to model_dag using outofrangeidx
            attentionlayer = -linearlayer.input_indecies_tuple[0]
            for hd in range(linearlayer.head - 1):  # 0 ~ head-1
                for attentionidx in range(attentionlayer):
                    model_dag.add_layer(
                        outofrange_idx + attentionidx + hd * attentionlayer,
                        True,  # NOTE Softmax is False
                    )
            concattuple = (
                idx - attentionlayer,
                outofrange_idx,
                linearlayer.head,
                attentionlayer,
            )
            concatlist.append(concattuple)

        # add edge
        if layertype != "CONCAT":
            (IIleft, IIright) = model_list[-1].input_indecies_tuple
            if IIleft < 0 and IIright < 0:
                model_dag.add_edge(idx + IIleft, idx)
                model_dag.add_edge(idx + IIright, idx)
            elif IIleft == 0 and IIright < 0:
                model_dag.add_edge(idx + IIright, idx)
            elif IIleft < 0 and IIright == 0:
                model_dag.add_edge(idx + IIleft, idx)
            else:
                model_dag.add_edge(-1, idx)
            """
            eidxint = (
                mxLCD_CNN["IDX"]
                if _modeltype == Model.ModelType.CNN
                else mxLCD_GEMM["IDX1"]
            )
            preidx = line[eidxint] + idx
            if preidx < idx:
                model_dag.add_edge(preidx, idx)
            else:
                model_dag.add_edge(-1, idx)  # -1 is a vNode for begin token

            if apdidx2_is_index2(layertype, line[eidxint + 1]):
                preidx2 = line[eidxint + 1] + idx
                assert preidx2 < idx
                model_dag.add_edge(preidx2, idx)
            """

        elif layertype == "CONCAT":
            #  add first head last layer edge to concat layer
            model_dag.add_edge(idx - 1, idx)
            # add left heads
            head = line[mxLCD_GEMM["M"]]
            attentionlayer = -line[mxLCD_GEMM["IDX1"]]
            head1begin_idx = idx - attentionlayer
            attentioninput_eidx = _model_nd[head1begin_idx, mxLCD_GEMM["IDX1"]]
            for hd in range(head - 1):
                headxbegin_idx = outofrange_idx + attentionlayer * hd
                if attentioninput_eidx == 0:
                    model_dag.add_edge(-1, headxbegin_idx)
                else:
                    model_dag.add_edge(
                        attentioninput_eidx + head1begin_idx, headxbegin_idx
                    )
                for atl in range(attentionlayer):
                    this_idx = headxbegin_idx + atl
                    for _idx in model_dag.toVertexDict[head1begin_idx + atl]:
                        that_idx = (
                            _idx - head1begin_idx + headxbegin_idx
                            if atl < attentionlayer - 1
                            else idx
                        )
                        model_dag.add_edge(this_idx, that_idx)
            outofrange_idx += attentionlayer * (head - 1)

    # assert model_dag.layernum == model_list.layernum
    # assert model_dag.layernum == _layernum
    if _modeltype == Model.ModelType.MHATTENTION:
        return model_dag, model_list, concatlist
    else:
        return model_dag, model_list


def get_idx_from_concat(idx, concatlist):
    # concattuple = (liststartidx, head2beginidx, head, attentionlayer)
    for tup in concatlist:
        if idx >= tup[1] and idx < tup[1] + (tup[2] - 1) * tup[3]:
            idx -= tup[1]
            while idx > tup[3]:
                idx -= tup[3]
            oidx = tup[0] + idx
        else:
            continue
    return oidx


def add_layerclass_to_dag(
    _modelDAG: Model.ModelDAG, _modelList: Model.ModelList, _concatlist: list
):
    modeltype = _modelDAG.modeltype
    layernumL = _modelList.layernum
    layernumG = _modelDAG.layernum
    assert layernumL == len(_modelList)
    if layernumL != layernumG and modeltype != Model.ModelType.MHATTENTION:
        print(
            "[Morax][System] generate queries fatal, layernum of List({}) and DAG({}) are different.".format(
                layernumL, layernumG
            )
        )
        raise DataError
    if _concatlist:
        _modelDAG.ConcatList = copy.deepcopy(_concatlist)
    for idx in _modelDAG.LayerIndexList:
        assert idx in _modelDAG.fromVertexDict and idx in _modelDAG.toVertexDict
        if idx < len(_modelList):
            _modelDAG.LayerClassDict[idx] = copy.deepcopy(_modelList[idx])
        else:
            oidx = get_idx_from_concat(idx, _concatlist)
            tmplayerclass = copy.deepcopy(_modelList[oidx])
            tmplayerclass.change_layerinfo(idx)
            _modelDAG.LayerClassDict[idx] = tmplayerclass
    return
