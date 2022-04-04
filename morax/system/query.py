# system query class
# Morax means the mixing of CMOS and RRAM
# Author: Tao Song.
# Last modification: 0316


from ast import If
from lzma import MODE_FAST
from sqlite3 import DataError
import queue
import sys
import numpy as np
import copy
import pandas as pd
import subprocess as SP

from sympy import Q
from morax.system.interface import *
from morax.model.layer import (
    LinearLayerType as LLT,
    NonlinearLayerType as NLT,
    mxLTD,
    mxNTD,
)
from morax.system.config import MoraxConfig, HWParam
from morax.hardware.buffer import DataBulk
from morax.hardware.chip import MoraxChip
import mapper
import math
from morax.model.model import ModelDAG, ModelList, ModelType
from morax.frontend.api import get_idx_from_concat, get_lookup_adress


# [bulk]
# indicate the data form of input and weight
# bulkfrom = (part or whole) feature: NCHW  weight: KCRS  MVM & GEMM: TODO
# dataname = W or F or TODO
# bulklabel = modelname_'L'+layeridx_'dataname'+bulkidx_bulksizeByte_bulkfrom

# [task]
# indicate the task excution using [output]
# taskform = (part or whole) batchN outputchannelC heighH widthW  MVM: outputdimO batchN GEMM: heightH widthW batchN
# CONV
# RRAM: CHWN onefeaturebyonefeature    OS: HWNC onechannelbyonechannel (FOR MAX KERNEL REUSE)
# TODO
# tasklabel = modelname_'L'+layeridx_'T'+taskidx_taskform


class VritualQuery:
    def __init__(self, _execution: VO):
        self.execution = _execution


class VritualQueryStart(VritualQuery):
    def __init__(self, _execution):
        super().__init__(_execution)


class VritualQueryMonitor(VritualQuery):
    def __init__(self, _execution, _thisidx, _thatidx, _actmod: str, _actlist=[]):
        super().__init__(_execution)
        self.thisidx = _thisidx
        self.thatidx = _thatidx
        self.actmod = _actmod
        self.actlist = copy.deepcopy(_actlist)


class QueryBuffer:
    def __init__(self, _databulkclass: DataBulk, _execution, _locationEnum, _toEnum):
        self.execution = _execution
        self.databulkclass = _databulkclass
        self.locationEnum = _locationEnum
        self.toEnum = _toEnum


class QueryExcute:
    def __init__(self, _layerclass, _tasklabel: str):
        self.tasklabel = _tasklabel
        self.layerclass = copy.deepcopy(_layerclass)
        self.taskfrom = self.get_taskform(_tasklabel)

    def get_taskform(self, _tasklabel):
        form = _tasklabel[_tasklabel.rfind("_") + 1 :]
        return form


class QueryExcuteOnTC(QueryExcute):
    def __init__(
        self,
        _layerclass,
        _tasklabel: str,
        _dfmod: str,
        _execution,
        _tasksizelist: list(tuple(int, int)),
        _bulksize,  # add 03.28
    ):
        super().__init__(_layerclass, _tasklabel)
        self.dfmod = _dfmod
        self.execution = _execution
        self.tasksizelist = _tasksizelist
        self.bulksize = _bulksize
        assert self.checkquery() is True

    def checkquery(self):
        dfsafe = self.dfmod == "OS" or self.dfmod == "Systolic"
        executionsafe = (
            self.execution in MoraxExecutionDict[ClusterComponent.TensorCore]
        )
        tasksafe = len(self.tasksizelist) <= MoraxConfig.PEArrayNum
        return dfsafe and executionsafe and tasksafe


class QueryExcuteOnNVTC(QueryExcute):
    def __init__(
        self,
        _layerclass,
        _tasklabel: str,
        _dfmod: str,
        _execution,
        _nvtcid: int,
        _clusterid: int,
        _sliceidlist,
        # _tasksize: tuple(int, int),
    ):
        super().__init__(_layerclass, _tasklabel)
        self.dfmod = _dfmod
        self.execution = _execution
        self.nvtcid = _nvtcid
        self.clusterid = _clusterid
        self.sliceidlist = _sliceidlist
        assert self.checkquery() is True

    def checkquery(self):
        dfsafe = self.dfmod == "Xbar" or self.dfmod == "LUT8" or self.dfmod == "LUT16"
        executionsafe = (
            self.execution in MoraxExecutionDict[ClusterComponent.nvTensorCore]
        )
        tasksafe = len(self.sliceidlist) <= MoraxConfig.RRAMSliceNum
        return dfsafe and executionsafe and tasksafe


class QueryExcuteOnVPU(QueryExcute):
    def __init__(
        self,
        _layerclass,
        _tasklabel: str,
        _dfmod: str,
        _execution,
        _tasksize: tuple(int, int) = (0, 0),
    ):
        super().__init__(_layerclass, _tasklabel)
        self.dfmod = _dfmod
        self.execution = _execution
        self.tasksize = _tasksize  # (rowparts, collines)
        assert self.checkquery() is True

    def checkquery(self):
        dfsafe = True if self.dfmod in ["Linear", "SoftMAX", "PostProcess"] else False
        executionsafe = self.execution in MoraxExecutionDict[ClusterComponent.VPU]
        tasksafe = True
        return dfsafe and executionsafe and tasksafe


class QueryExcuteOnSMU(QueryExcute):
    def __init__(
        self,
        _layerclass,  # Empty
        _tasklabel: str,
        _dfmod: str,
        _execution,
        _tasksize: tuple(int, int) = (0, 0),
    ):
        super().__init__(_layerclass, _tasklabel)
        self.dfmod = _dfmod
        self.execution = _execution
        self.tasksize = _tasksize
        assert self.checkquery() is True

    def checkquery(self):
        dfsafe = self.dfmod == "UpStream" or self.dfmod == "DownStream"
        executionsafe = self.execution in MoraxExecutionDict[ClusterComponent.SMU]
        tasksafe = True
        return dfsafe and executionsafe and tasksafe


class QueryRingBus:
    def __init__(self, _databulkclass: DataBulk, _fromCluster, _toCluster):
        self.databulkclass = _databulkclass
        self.fromCluster = _fromCluster
        self.toCluster = _toCluster


class QueryDMA:
    def __init__(self, _databulkclass: DataBulk, _toCluster):
        self.databulkclass = _databulkclass
        self.toCluster = _toCluster


""" ========================================================================================================================
                     generate layer query and compilie them to sub query
============================================================================================================================
"""


class LayerQuery:
    def __init__(
        self,
        _q_index: int,
        _batch: int,
        _layerclass,
        _assignment,
        _indegree,
        _outdegree,
    ) -> None:
        self.q_index = _q_index
        self.batch = _batch
        self.assignment = _assignment
        self.layerclass = copy.deepcopy(_layerclass)
        self.iodegree = {"in": _indegree, "out": _outdegree}
        self.SubQueryList = []

        # assignment: listoftuple_clstid_nvtcid_sliceidlist
        # default []
        if isinstance(self.assignment, list):
            assert mapper.check_mapping_with_query(
                self.layerclass.layer_index
            ), "{}".format(self.layerclass.layer_name)

    def compile(self, _modelname, moraxchip: MoraxChip, concatlist=[]):
        # Generate subqueries of this layer query
        print("[Morax][System] Compiling Query {}.".format(self.q_index))
        layertype = self.layerclass.layer_type
        if layertype in LLT:
            if layertype == LLT.CONCAT:
                self.SubQueryList = copy.deepcopy(
                    compileCONCAT(self.q_index, self.layerclass, concatlist)
                )
            else:
                if self.assignment:  # on NVTC
                    self.SubQueryList = copy.deepcopy(
                        compileRRAM(
                            self.q_index,
                            _modelname,
                            self.layerclass,
                            self.assignment,
                            moraxchip,
                            self.batch,
                            self.iodegree["out"],
                        )
                    )
                else:
                    self.SubQueryList = copy.deepcopy(
                        compileCMOS(
                            self.q_index,
                            _modelname,
                            self.layerclass,
                            moraxchip,
                            self.batch,
                            self.iodegree["out"],
                        )
                    )
        elif layertype in NLT:
            self.SubQueryList = copy.deepcopy(compileVPU())
        return


def generate_queries(
    _modelList: ModelList,
    _modelDAG: ModelDAG,
    _moraxchip: MoraxChip,
    concatlist=[],
    _batch=1,
):
    modeltype = ModelDAG.modeltype
    modelname = ModelDAG.modelname
    layernumL = _modelList.layernum
    layernumG = _modelDAG.layernum
    if layernumL != layernumG and modeltype != ModelType.MHATTENTION:
        print(
            "[Morax][System] generate queries fatal, layernum of List({}) and DAG({}) are different.".format(
                layernumL, layernumG
            )
        )
        raise DataError
    totalquery = 0
    for idx in _modelDAG.LayerIndexList:
        assert idx in _modelDAG.fromVertexDict and idx in _modelDAG.toVertexDict
        if idx < len(_modelList):
            q = LayerQuery(
                idx,
                _batch,
                _modelList[idx],
                _modelDAG.LayerAssignmentDict[idx],
                len(_modelDAG.fromVertexDict[idx]),
                len(_modelDAG.toVertexDict[idx]),
            )
        else:
            oidx = get_idx_from_concat(idx, concatlist)
            q = LayerQuery(
                idx,
                _batch,
                _modelList[oidx],  # TODO: CHANGE INDEX TUPLE of layerclass
                _modelDAG.LayerAssignmentDict[idx],
                len(_modelDAG.fromVertexDict[idx]),
                len(_modelDAG.toVertexDict[idx]),
            )
        q.compile(modelname, _moraxchip, concatlist)
        _modelDAG.LayerQueryClassDict[idx] = copy.deepcopy(q)
        totalquery += 1
    assert totalquery == _modelDAG.layernum

    # NOTE DAG begins with virtual layer -1
    # Branch conds:
    # 1 Residual -> this cluster
    # 2 CONCAT multi-head -> one head one cluster
    # 3 CONCAT group or next multi-path ->try to use less clusters


def generate_queries_mt(
    _modelList1,
    _modelList2,
    _modelDAG1,
    _modelDAG2,
    _modeltype1,
    _modeltype2,
    _assignmentList1,
    _assignmentList2,
    _batch=1,
):
    layernum1 = len(_modelList1)
    layernum2 = len(_modelList2)
    assert layernum1 == len(_assignmentList1)
    assert layernum2 == len(_assignmentList2)
    return


def make_tasklabel(mn, li, ti, bf: str):
    return mn + "_L" + str(li) + "T" + str(ti) + "_" + bf


ALL = 114514


def compileCONCAT(_index, _layerclass, _concatlist):
    #  list of concattuple = (liststartidx, head2beginidx, head, attentionlayer)
    VSubQueryList = []
    for tup in _concatlist:
        if _index == tup[3] + tup[0]:
            this_concattup = copy.deepcopy(tup)
    assert this_concattup
    heads = this_concattup[2]
    assert heads == _layerclass.head
    for hd in range(heads):
        if hd == 0:
            vqem = VritualQueryMonitor(VO.EditMemonitor, _index - 1, _index, "CONCAT")
            VSubQueryList.append(vqem)
        else:
            this_index = this_concattup[1] * (hd - 1) + this_concattup[3] - 1
            vqem = VritualQueryMonitor(VO.EditMemonitor, this_index, _index, "CONCAT")
            VSubQueryList.append(vqem)
    return VSubQueryList


def compileRRAM(
    _index, _modelname, _layerclass, _doclotnsl: dict, _chip: MoraxChip, _batch, token
):
    # NOTE BATCH IS ALWAYS THE LAST
    # dict of clstid: [tuple1(nvtcid, sliceidlist), tuple2, ... ]
    SubQueryList = []
    layertype = _layerclass.layer_type
    (IIleft, IIright) = _layerclass.input_indecies_tuple
    if layertype == LLT.Linear or layertype == LLT.VMM:
        M = math.ceil(_layerclass.row_dim * 1.0 / MoraxConfig.RRAMXbarSize)
        N = math.ceil(_layerclass.col_dim * 1.0 / MoraxConfig.RRAMXbarSize)
        M_tail = _layerclass.row_dim % MoraxConfig.RRAMXbarSize
        N_tail = _layerclass.col_dim % MoraxConfig.RRAMXbarSize
        taskindex = -1
        for bat in range(_batch):
            for clstid, nvtclist in _doclotnsl.items():
                for nvtctup in nvtclist:
                    (nvtcid, sliceidlist) = nvtctup
                    mapinfo = []
                    for info in sliceidlist:
                        mapinfo.append(
                            _chip.ClusterList[clstid]
                            .nvTensorCoreList[nvtcid]
                            .RRAMSliceObjList[info]
                            .layerinfo[1]
                        )
                    bulkinfo = []
                    for tup in mapinfo:
                        if tup[0] not in bulkinfo:
                            bulkinfo.append(tup[0])
                    # make query
                    datatype = "FTR" if layertype == LLT.Linear else "VEC"
                    bulkscratch = {}
                    taskindex += 1
                    bulkscratch["B"] = bat
                    if mxLTD[layertype] == "Linear":
                        bulkscratch["HW"] = ALL
                        rowpart = "C"
                    else:
                        bulkscratch["N"] = ALL
                        rowpart = "M"
                    bulkscratch[rowpart] = []
                    bs = 0
                    for inf in bulkinfo:
                        if inf == M and M_tail > 0:
                            bs += M_tail
                        else:
                            bs += MoraxConfig.RRAMXbarSize
                    bulkscratch[rowpart].append(inf)
                    bs = bs * MoraxConfig.PrecisionBits / 8
                    bulk = DataBulk(
                        _modelname, _index + +IIleft, datatype, bs, bulkscratch
                    )
                    qr = QueryBuffer(bulk, BO.Read, CC.FeatureBuffer, CC.nvTensorCore)
                    tasklabel = make_tasklabel(
                        _modelname, _index, taskindex, mxLTD[layertype]
                    )
                    qe = QueryExcuteOnNVTC(
                        _layerclass,
                        tasklabel,
                        "Xbar",
                        LLT.Linear,
                        nvtcid,
                        clstid,
                        sliceidlist,
                    )
                    SubQueryList.append(copy.deepcopy(qr))
                    SubQueryList.append(copy.deepcopy(qe))
            tasklabel = make_tasklabel(_modelname, _index, 0, "PostProcess")
            qv = QueryExcuteOnVPU(
                _layerclass,
                tasklabel,
                "PostProcess",
                LLT.Linear,
                (M, _layerclass.col_dim),
            )
            SubQueryList.append(copy.deepcopy(qv))
            if mxLTD[layertype] == "Linear":
                bulk = DataBulk(
                    _modelname,
                    _index,
                    "FTR",
                    _layerclass.col_dim * MoraxConfig.PrecisionBits / 8,
                    {"B": bat, "HW": ALL, "C": ALL},
                    token,
                )
            else:
                bulk = DataBulk(
                    _modelname,
                    _index,
                    "VEC",
                    _layerclass.col_dim * MoraxConfig.PrecisionBits / 8,
                    {"B": bat, "M": ALL, "N": ALL},
                    token,
                )
            qw = QueryBuffer(bulk, BO.Write, CC.VPU, CC.FeatureBuffer)
            SubQueryList.append(copy.deepcopy(qw))
        # End for

    if layertype == LLT.CONV:
        M = math.ceil(
            _layerclass.in_channel
            * _layerclass.kernel_size ** 2
            * 1.0
            / MoraxConfig.RRAMXbarSize
        )
        N = math.ceil(_layerclass.out_channel * 1.0 / MoraxConfig.RRAMXbarSize)
        M_tail = (
            _layerclass.in_channel
            * _layerclass.kernel_size ** 2
            % MoraxConfig.RRAMXbarSize
        )
        N_tail = _layerclass.out_channel % MoraxConfig.RRAMXbarSize
        rram_taskindex = -1
        vpu_taskindex = -1
        for bat in range(_batch):
            omapsize = _layerclass.feature_size / _layerclass.stride
            for row_iter in range(omapsize):
                for col_iter in range(omapsize):
                    # make simple bulk
                    datatype = "FTR"
                    bulkscratch = {}
                    bulkscratch["B"] = bat
                    bulkscratch["HW"] = col_iter + row_iter * omapsize
                    bulkscratch["C"] = ALL
                    bs = (
                        _layerclass.kernel_size
                        * _layerclass.in_channel
                        * MoraxConfig.PrecisionBits
                        / 8
                    )
                    bs *= (
                        _layerclass.kernel_size if col_iter == 0 else _layerclass.stride
                    )
                    bulk = DataBulk(
                        _modelname, _index + IIleft, datatype, bs, bulkscratch
                    )
                    qr = QueryBuffer(bulk, BO.Read, CC.FeatureBuffer, CC.nvTensorCore)
                    SubQueryList.append(copy.deepcopy(qr))
                    if IIright != 0:
                        bulk2 = DataBulk(
                            _modelname, _index + IIright, datatype, bs, bulkscratch
                        )
                        qr2 = QueryBuffer(
                            bulk2, BO.Read, CC.FeatureBuffer, CC.nvTensorCore
                        )
                        SubQueryList.append(copy.deepcopy(qr2))
                    # make subexe query
                    # dict of clstid: [tuple1(nvtcid, sliceidlist), tuple2, ... ]
                    for clstid, nvtclist in _doclotnsl.items():
                        for nvtctup in nvtclist:
                            (nvtcid, sliceidlist) = nvtctup
                            rram_taskindex += 1
                            rtasklabel = make_tasklabel(
                                _modelname, _index, rram_taskindex, "CONV"
                            )
                            qe = QueryExcuteOnNVTC(
                                _layerclass,
                                rtasklabel,
                                "Xbar",
                                LLT.CONV,
                                nvtcid,
                                clstid,
                                sliceidlist,
                            )
                            SubQueryList.append(copy.deepcopy(qe))
                    vpu_taskindex += 1
                    vtasklabel = make_tasklabel(
                        _modelname, _index, vpu_taskindex, "PostProcess"
                    )
                    qv = QueryExcuteOnVPU(
                        _layerclass,
                        vtasklabel,
                        "PostProcess",
                        LLT.CONV,
                        (M, _layerclass.out_channel),
                    )
                    SubQueryList.append(copy.deepcopy(qv))
                    # make writeback bulk
                    bulk = DataBulk(
                        _modelname,
                        _index,
                        "FTR",
                        _layerclass.out_channel * MoraxConfig.PrecisionBits / 8,
                        {"B": bat, "HW": col_iter + row_iter * omapsize, "C": ALL},
                        token,
                    )
                    qw = QueryBuffer(bulk, BO.Write, CC.VPU, CC.FeatureBuffer)
                    SubQueryList.append(copy.deepcopy(qw))
            # End for

    if layertype == LLT.TRCONV:
        M = math.ceil(
            _layerclass.in_channel
            * _layerclass.kernel_size ** 2
            * 1.0
            / MoraxConfig.RRAMXbarSize
        )
        N = math.ceil(_layerclass.out_channel * 1.0 / MoraxConfig.RRAMXbarSize)
        M_tail = (
            _layerclass.in_channel
            * _layerclass.kernel_size ** 2
            % MoraxConfig.RRAMXbarSize
        )
        N_tail = _layerclass.out_channel % MoraxConfig.RRAMXbarSize
        rram_taskindex = -1
        vpu_taskindex = -1
        for bat in range(_batch):
            omapsize = (
                _layerclass.feature_size - 1
            ) * _layerclass.stride + _layerclass.kernel_size
            for row_iter in range(omapsize):
                for col_iter in range(omapsize):
                    if (
                        (
                            (row_iter + _layerclass.kernel_size - 1)
                            - (_layerclass.kernel_size - 1)
                        )
                        % _layerclass.stride
                        == 0
                        and col_iter % _layerclass.stride == 0
                    ):
                        # make simple bulk
                        datatype = "FTR"
                        bulkscratch = {}
                        bulkscratch["B"] = bat
                        bulkscratch["HW"] = (col_iter / _layerclass.stride - 1) + (
                            row_iter / _layerclass.stride - 1
                        ) * _layerclass.feature_size
                        bulkscratch["C"] = ALL
                        bs = _layerclass.in_channel * MoraxConfig.PrecisionBits / 8
                        bulk = DataBulk(
                            _modelname, _index + IIleft, datatype, bs, bulkscratch
                        )
                        qr = QueryBuffer(
                            bulk, BO.Read, CC.FeatureBuffer, CC.nvTensorCore
                        )
                        SubQueryList.append(copy.deepcopy(qr))
                    # make subexe query
                    # dict of clstid: [tuple1(nvtcid, sliceidlist), tuple2, ... ]
                    for clstid, nvtclist in _doclotnsl.items():
                        for nvtctup in nvtclist:
                            (nvtcid, sliceidlist) = nvtctup
                            rram_taskindex += 1
                            rtasklabel = make_tasklabel(
                                _modelname, _index, rram_taskindex, "TRCONV"
                            )
                            qe = QueryExcuteOnNVTC(
                                _layerclass,
                                rtasklabel,
                                "Xbar",
                                LLT.CONV,
                                nvtcid,
                                clstid,
                                sliceidlist,
                            )
                            SubQueryList.append(copy.deepcopy(qe))
                    vpu_taskindex += 1
                    vtasklabel = make_tasklabel(
                        _modelname, _index, vpu_taskindex, "PostProcess"
                    )
                    qv = QueryExcuteOnVPU(
                        _layerclass,
                        vtasklabel,
                        "PostProcess",
                        LLT.TRCONV,
                        (M, _layerclass.out_channel),
                    )
                    SubQueryList.append(copy.deepcopy(qv))
                    # make writeback bulk
                    bulk = DataBulk(
                        _modelname,
                        _index,
                        "FTR",
                        _layerclass.out_channel * MoraxConfig.PrecisionBits / 8,
                        {"B": bat, "HW": col_iter + row_iter * omapsize, "C": ALL},
                        token,
                    )
                    qw = QueryBuffer(bulk, BO.Write, CC.VPU, CC.FeatureBuffer)
                    SubQueryList.append(copy.deepcopy(qw))
            # End for

    if layertype == LLT.NGCONV:
        M = math.ceil(
            (_layerclass.in_channel / _layerclass.group)
            * _layerclass.kernel_size ** 2
            * 1.0
            / MoraxConfig.RRAMXbarSize
        )
        N = math.ceil(
            (_layerclass.out_channel / _layerclass.group)
            * 1.0
            / MoraxConfig.RRAMXbarSize
        )
        M_tail = (
            (_layerclass.in_channel / _layerclass.group)
            * _layerclass.kernel_size ** 2
            / MoraxConfig.RRAMXbarSize
        )
        N_tail = (
            _layerclass.out_channel / _layerclass.group
        ) / MoraxConfig.RRAMXbarSize
        rram_taskindex = -1
        vpu_taskindex = -1

        for bat in range(_batch):
            for grp in range(_layerclass.group):
                omapsize = _layerclass.feature_size / _layerclass.stride
                for row_iter in range(omapsize):
                    for col_iter in range(omapsize):
                        # make simple bulk
                        datatype = "FTR"
                        bulkscratch = {}
                        bulkscratch["B"] = bat
                        bulkscratch["HW"] = col_iter + row_iter * omapsize
                        bulkscratch["C"] = grp
                        bs = (
                            _layerclass.kernel_size
                            * _layerclass.in_channel
                            * (MoraxConfig.PrecisionBits / 8)
                            / _layerclass.group
                        )
                        bs *= (
                            _layerclass.kernel_size
                            if col_iter == 0
                            else _layerclass.stride
                        )
                        bulk = DataBulk(
                            _modelname, _index + IIleft, datatype, bs, bulkscratch
                        )
                        qr = QueryBuffer(
                            bulk, BO.Read, CC.FeatureBuffer, CC.nvTensorCore
                        )
                        SubQueryList.append(copy.deepcopy(qr))
                        # make subexe query
                        # NOTE dict of group: [tuple1(clstid, nvtcid, sliceidlist), tuple2, ... ]
                        grpdict = _doclotnsl[grp]
                        for (clstid, nvtcid, sliceidlist) in grpdict:
                            rram_taskindex += 1
                            rtasklabel = make_tasklabel(
                                _modelname, _index, rram_taskindex, "NGCONV"
                            )
                            qe = QueryExcuteOnNVTC(
                                _layerclass,
                                rtasklabel,
                                "Xbar",
                                LLT.NGCONV,
                                nvtcid,
                                clstid,
                                sliceidlist,
                            )
                            SubQueryList.append(copy.deepcopy(qe))
                        vpu_taskindex += 1
                        vtasklabel = make_tasklabel(
                            _modelname, _index, vpu_taskindex, "PostProcess"
                        )
                        qv = QueryExcuteOnVPU(
                            _layerclass,
                            vtasklabel,
                            "PostProcess",
                            LLT.NGCONV,
                            (M, _layerclass.out_channel / _layerclass.group),
                        )
                        SubQueryList.append(copy.deepcopy(qv))
                        # make writeback bulk
                        bulk = DataBulk(
                            _modelname,
                            _index,
                            "FTR",
                            _layerclass.out_channel
                            / _layerclass.group
                            * (MoraxConfig.PrecisionBits / 8),
                            {"B": bat, "HW": col_iter + row_iter * omapsize, "C": grp},
                            token,
                        )
                        qw = QueryBuffer(bulk, BO.Write, CC.VPU, CC.FeatureBuffer)
                        SubQueryList.append(copy.deepcopy(qw))
        # End for

    if layertype == LLT.GEMM:
        # idxtup = _layerclass.input_indecies_tuple
        if IIright == 0:
            iidx = IIleft
            row_dim = _layerclass.k_dim
            col_dim = _layerclass.n_dim
            v_dim = _layerclass.m_dim
        else:
            iidx = IIright
            row_dim = _layerclass.k_dim
            col_dim = _layerclass.m_dim
            v_dim = _layerclass.n_dim
        M = math.ceil(row_dim * 1.0 / MoraxConfig.RRAMXbarSize)
        N = math.ceil(col_dim * 1.0 / MoraxConfig.RRAMXbarSize)
        M_tail = row_dim % MoraxConfig.RRAMXbarSize
        N_tail = col_dim % MoraxConfig.RRAMXbarSize
        # assume fetch batch is 8
        vfetch = 16
        V = math.ceil(v_dim * 1.0 / vfetch)
        rram_taskindex = -1
        vpu_taskindex = -1
        for bat in range(_batch):
            for vf in range(V):
                for clstid, nvtclist in _doclotnsl.items():
                    for nvtctup in nvtclist:
                        (nvtcid, sliceidlist) = nvtctup
                        mapinfo = []
                        for info in sliceidlist:
                            mapinfo.append(
                                _chip.ClusterList[clstid]
                                .nvTensorCoreList[nvtcid]
                                .RRAMSliceObjList[info]
                                .layerinfo[1]
                            )
                        bulkinfo = []
                        for tup in mapinfo:
                            if tup[0] not in bulkinfo:
                                bulkinfo.append(tup[0])
                        # make query
                        datatype = "MAT"
                        bulkscratch = {}
                        bulkscratch["B"] = bat
                        bulkscratch["M"] = vf
                        bulkscratch["N"] = []
                        bs = 0
                        for inf in bulkinfo:
                            if inf == M and M_tail > 0:
                                bs += M_tail
                            else:
                                bs += MoraxConfig.RRAMXbarSize
                        bulkscratch["N"].append(inf)
                        bs = vfetch * bs * MoraxConfig.PrecisionBits / 8
                        bulk = DataBulk(
                            _modelname, _index + iidx, datatype, bs, bulkscratch
                        )
                        qr = QueryBuffer(
                            bulk, BO.Read, CC.FeatureBuffer, CC.nvTensorCore
                        )
                        SubQueryList.append(copy.deepcopy(qr))

                for line in range(vfetch):
                    for clstid, nvtclist in _doclotnsl.items():
                        for nvtctup in nvtclist:
                            rram_taskindex += 1
                            tasklabel = make_tasklabel(
                                _modelname, _index, rram_taskindex, "GEMM"
                            )
                            qe = QueryExcuteOnNVTC(
                                _layerclass,
                                tasklabel,
                                "Xbar",
                                LLT.GEMM,
                                nvtcid,
                                clstid,
                                sliceidlist,
                            )
                            SubQueryList.append(copy.deepcopy(qe))
                    vpu_taskindex += 1
                    vtasklabel = make_tasklabel(
                        _modelname, vpu_taskindex, 0, "PostProcess"
                    )
                    qv = QueryExcuteOnVPU(
                        _layerclass,
                        vtasklabel,
                        "PostProcess",
                        LLT.GEMM,
                        (M, _layerclass.col_dim),
                    )
                    SubQueryList.append(copy.deepcopy(qv))
                    bulk = DataBulk(
                        _modelname,
                        _index,
                        "MAT",
                        _layerclass.col_dim * MoraxConfig.PrecisionBits / 8,
                        {"B": bat, "M": line, "N": ALL},
                        token,
                    )
                    qw = QueryBuffer(bulk, BO.Write, CC.VPU, CC.FeatureBuffer)
                    SubQueryList.append(copy.deepcopy(qw))
            # End for
    return SubQueryList


def compileCMOS(_index, _modelname, _layerclass, _chip: MoraxChip, _batch, token):
    # NOTE BATCH IS ALWAYS assigned on one same tc
    SubQueryList = []
    layertype = _layerclass.layer_type
    onetcsize = MoraxConfig.PEArraySize * MoraxConfig.PEArrayNum
    (IIleft, IIright) = _layerclass.input_indecies_tuple
    ICtypeLeft = (
        CC.FeatureBuffer if _layerclass.input_indecies_tuple[0] < 0 else CC.WeightBuffer
    )
    ICtypeRight = (
        CC.FeatureBuffer if _layerclass.input_indecies_tuple[1] < 0 else CC.WeightBuffer
    )
    cmos_taskindex = -1
    vpu_taskindex = -1
    # go
    if layertype == LLT.Linear or layertype == LLT.VMM:
        # for: M N B
        M = math.ceil(_layerclass.row_dim * 1.0 / MoraxConfig.PEArraySize)
        N = math.ceil(_layerclass.col_dim * 1.0 / onetcsize)
        N_tail = _layerclass.col_dim % onetcsize
        B = _batch
        for m in range(M):
            # make left op bulk
            fbulkscratch = {}
            if layertype == LLT.Linear:
                datatype = "FTR"
                fbulkscratch["B"] = ALL
                fbulkscratch["HW"] = ALL
                fbulkscratch["C"] = m
            else:
                datatype = "VEC"
                fbulkscratch["B"] = ALL
                fbulkscratch["M"] = m
                fbulkscratch["N"] = ALL
            fbulk = DataBulk(
                _modelname,
                _index + IIleft,
                datatype,
                B * MoraxConfig.PEArraySize * MoraxConfig.PrecisionBits / 8,
                fbulkscratch,
            )
            qrf = QueryBuffer(fbulk, BO.Read, ICtypeLeft, CC.TensorCore)
            SubQueryList.append(copy.deepcopy(qrf))
            # make right op bulk
            for n in range(N):
                wbulkscratch = {}
                if layertype == LLT.Linear:
                    datatype = "WET"
                    wbulkscratch["K"] = m
                    wbulkscratch["C"] = n
                    wbulkscratch["RS"] = ALL
                else:
                    datatype = "MAT"
                    wbulkscratch["B"] = ALL
                    wbulkscratch["M"] = m
                    wbulkscratch["N"] = n
                wbulk = DataBulk(
                    _modelname,
                    _index + IIright,
                    datatype,
                    MoraxConfig.PEArraySize * onetcsize * MoraxConfig.PrecisionBits / 8,
                    wbulkscratch,
                )
                qrw = QueryBuffer(wbulk, BO.Read, ICtypeRight, CC.TensorCore)
                SubQueryList.append(copy.deepcopy(qrw))
                # make query
                for b in range(B):
                    cmos_taskindex += 1
                    ctasklabel = make_tasklabel(
                        _modelname, _index, cmos_taskindex, mxLTD[layertype]
                    )
                    if n == N - 1 and N_tail > 0:
                        tasksize_list = copy.deepcopy(make_tc_task_list(N_tail))
                    else:
                        tasksize_list = [
                            (MoraxConfig.PEArraySize, MoraxConfig.PEArraySize)
                        ] * MoraxConfig.PEArrayNum
                    tcbulksize = MoraxConfig.PEArraySize * MoraxConfig.PrecisionBits / 8
                    if b == 0:
                        tcbulksize += (
                            sum(tasksize_list)
                            * MoraxConfig.PEArraySize
                            * MoraxConfig.PrecisionBits
                            / 8
                        )
                    qe = QueryExcuteOnTC(
                        _layerclass,
                        ctasklabel,
                        "Systolic",
                        layertype,
                        tasksize_list,
                        tcbulksize,
                    )
                    SubQueryList.append(copy.deepcopy(qe))
        # make vpu query
        for b in range(B):
            vtasklabel = make_tasklabel(_modelname, _index, b, "PostProcess")
            qv = QueryExcuteOnVPU(
                _layerclass,
                vtasklabel,
                "PostProcess",
                layertype,
                (M, _layerclass.col_dim),
            )
            SubQueryList.append(copy.deepcopy(qv))
            if layertype == LLT.Linear:
                wbbulk = DataBulk(
                    _modelname,
                    _index,
                    "FTR",
                    _layerclass.col_dim * MoraxConfig.PrecisionBits / 8,
                    {"B": b, "HW": ALL, "C": ALL},
                    token,
                )
            else:
                wbbulk = DataBulk(
                    _modelname,
                    _index,
                    "VEC",
                    _layerclass.col_dim * MoraxConfig.PrecisionBits / 8,
                    {"B": b, "M": ALL, "N": ALL},
                    token,
                )
            qw = QueryBuffer(wbbulk, BO.Write, CC.VPU, CC.FeatureBuffer)
            SubQueryList.append(copy.deepcopy(qw))
        # End for

    if layertype == LLT.CONV:
        # for: C B
        H = math.ceil(
            _layerclass.feature_size
            / _layerclass.stride
            * 1.0
            / MoraxConfig.PEArraySize
        )
        W = math.ceil(
            _layerclass.feature_size
            / _layerclass.stride
            * 1.0
            / MoraxConfig.PEArraySize
        )
        C = math.ceil(
            _layerclass.out_channel * 1.0 * H * W / MoraxConfig.PEArrayNum
        )  # ONETC
        H_tail = _layerclass.feature_size % MoraxConfig.PEArraySize
        W_tail = H_tail
        B = _batch
        chnum = 0
        for comp in range(C):
            # spcify task
            tasksize_listoftup = [] * MoraxConfig.PEArrayNum
            this_ch = comp * MoraxConfig.PEArrayNum / (H * W)
            that_ch = (comp + 1) * MoraxConfig.PEArrayNum / (H * W)
            hwidbegin = comp * MoraxConfig.PEArrayNum % (H * W)
            for tmptid in range(MoraxConfig.PEArrayNum):
                tmptup = []
                if this_ch == chsize:
                    tmptup[1] = 0
                    tmptup[2] = 0
                else:
                    ww = hwidbegin % W
                    hh = hwidbegin / W
                    if ww == W - 1 and W_tail > 0:
                        tmptup[1] = W_tail
                    else:
                        tmptup[1] = MoraxConfig.PEArraySize
                    if hh == H - 1 and H_tail > 0:
                        tmptup[0] = H_tail
                    else:
                        tmptup[0] = MoraxConfig.PEArraySize
                tasksize_listoftup[tmptid] = tuple(tmptup)
                hwidbegin += 1
                if (comp * MoraxConfig.PEArrayNum + hwidbegin) / (H * W) > this_ch:
                    assert this_ch < that_ch
                    hwidbegin = 0
                    this_ch += 1
                    chnum += 1
            # make weight op bulk
            wbulkscratch = {}
            datatype = "WET"
            wbulkscratch["K"] = ALL
            wbulkscratch["RS"] = ALL
            wbulkscratch["C"] = comp
            wbulksize = (
                _layerclass.kernel_size ** 2
                * _layerclass.in_channel
                * chnum
                * MoraxConfig.PrecisionBits
                / 8
            )
            wbulk = DataBulk(_modelname, _index, datatype, wbulksize, wbulkscratch,)
            qrw = QueryBuffer(wbulk, BO.Read, CC.WeightBuffer, CC.TensorCore)
            SubQueryList.append(copy.deepcopy(qrw))
            # make feature op bulk
            for bat in range(B):
                fbulkscratch = {}
                datatype = "FTR"
                fbulkscratch["B"] = bat
                fbulkscratch["C"] = ALL
                fbulkscratch["HW"] = (
                    comp * MoraxConfig.PEArrayNum % (H * W) / MoraxConfig.PEArrayNum
                    if H * W >= MoraxConfig.PEArrayNum
                    else ALL
                )
                finsize2d = 0
                foutsize2d = 0
                if H * W >= MoraxConfig.PEArrayNum:
                    for tup in tasksize_listoftup:
                        finsize2d += (tup[0] + _layerclass.kernel_size - 1) * (
                            tup[1] + _layerclass.kernel_size - 1
                        )
                        foutsize2d += tup[0] * tup[1]
                else:
                    finsize2d = _layerclass.feature_size ** 2
                    foutsize2d = (_layerclass.feature_size / _layerclass.stride) ** 2
                # finsize2d = (fsize[0] + _layerclass.kernel_size - 1) * (
                #    fsize[1] + _layerclass.kernel_size - 1)
                inch = (
                    _layerclass.in_channel
                    if IIright == 0
                    else _layerclass.in_channel / 2
                )
                fbulk = DataBulk(
                    _modelname,
                    _index + IIleft,
                    datatype,
                    finsize2d * inch * MoraxConfig.PrecisionBits / 8,
                    fbulkscratch,
                )
                qrf = QueryBuffer(fbulk, BO.Read, ICtypeLeft, CC.TensorCore)
                SubQueryList.append(copy.deepcopy(qrf))
                if IIright != 0:
                    fbulk2 = DataBulk(
                        _modelname,
                        _index + IIright,
                        datatype,
                        finsize2d * inch * MoraxConfig.PrecisionBits / 8,
                        fbulkscratch,
                    )
                qrf2 = QueryBuffer(fbulk, BO.Read, ICtypeRight, CC.TensorCore)
                SubQueryList.append(copy.deepcopy(qrf2))
                # make query
                cmos_taskindex += 1
                ctasklabel = make_tasklabel(
                    _modelname, _index, cmos_taskindex, mxLTD[layertype]
                )
                tcbulksize = (
                    finsize2d * _layerclass.in_channel * MoraxConfig.PrecisionBits / 8
                )
                if bat == 0:
                    tcbulksize += wbulksize
                qe = QueryExcuteOnTC(
                    _layerclass,
                    ctasklabel,
                    "OS",
                    layertype,
                    tasksize_listoftup,
                    tcbulksize,
                )
                SubQueryList.append(copy.deepcopy(qe))
                # make writeback query
                # foutsize2d = fsize[0] * fsize[1] * MoraxConfig.PrecisionBits / 8
                wbbulk = DataBulk(
                    _modelname,
                    _index,
                    "FTR",
                    foutsize2d * MoraxConfig.PrecisionBits / 8,
                    {"B": bat, "HW": fbulkscratch["HW"], "C": comp},
                    token,
                )
                qw = QueryBuffer(wbbulk, BO.Write, CC.TensorCore, CC.FeatureBuffer)
                SubQueryList.append(copy.deepcopy(qw))
        # End for

    if layertype == LLT.DWCONV:
        # for: C B
        H = math.ceil(
            _layerclass.feature_size
            / _layerclass.stride
            * 1.0
            / MoraxConfig.PEArraySize
        )
        W = math.ceil(
            _layerclass.feature_size
            / _layerclass.stride
            * 1.0
            / MoraxConfig.PEArraySize
        )
        C = math.ceil(
            _layerclass.channel * 1.0 * H * W / MoraxConfig.PEArrayNum
        )  # ONETC
        H_tail = _layerclass.feature_size % MoraxConfig.PEArraySize
        W_tail = H_tail
        B = _batch
        chnum = 0
        for comp in range(C):
            # spcify task
            tasksize_listoftup = [] * MoraxConfig.PEArrayNum
            this_ch = comp * MoraxConfig.PEArrayNum / (H * W)
            that_ch = (comp + 1) * MoraxConfig.PEArrayNum / (H * W)
            hwidbegin = comp * MoraxConfig.PEArrayNum % (H * W)
            for tmptid in range(MoraxConfig.PEArrayNum):
                tmptup = []
                if this_ch == chsize:
                    tmptup[1] = 0
                    tmptup[2] = 0
                else:
                    ww = hwidbegin % W
                    hh = hwidbegin / W
                    if ww == W - 1 and W_tail > 0:
                        tmptup[1] = W_tail
                    else:
                        tmptup[1] = MoraxConfig.PEArraySize
                    if hh == H - 1 and H_tail > 0:
                        tmptup[0] = H_tail
                    else:
                        tmptup[0] = MoraxConfig.PEArraySize
                tasksize_listoftup[tmptid] = tuple(tmptup)
                hwidbegin += 1
                if (comp * MoraxConfig.PEArrayNum + hwidbegin) / (H * W) > this_ch:
                    assert this_ch < that_ch
                    hwidbegin = 0
                    this_ch += 1
                    chnum += 1
            # make weight op bulk
            wbulkscratch = {}
            datatype = "WET"
            wbulkscratch["K"] = ALL
            wbulkscratch["RS"] = ALL
            wbulkscratch["C"] = comp
            wbulksize = (
                _layerclass.kernel_size ** 2
                # * _layerclass.in_channel
                * chnum
                * MoraxConfig.PrecisionBits
                / 8
            )
            wbulk = DataBulk(_modelname, _index, datatype, wbulksize, wbulkscratch)
            qrw = QueryBuffer(wbulk, BO.Read, CC.WeightBuffer, CC.TensorCore)
            SubQueryList.append(copy.deepcopy(qrw))
            # make feature op bulk
            for bat in range(B):
                fbulkscratch = {}
                datatype = "FTR"
                fbulkscratch["B"] = bat
                fbulkscratch["C"] = comp
                fbulkscratch["HW"] = (
                    comp * MoraxConfig.PEArrayNum % (H * W) / MoraxConfig.PEArrayNum
                    if H * W >= MoraxConfig.PEArrayNum
                    else ALL
                )
                finsize2d = 0
                foutsize2d = 0
                if H * W >= MoraxConfig.PEArrayNum:
                    for tup in tasksize_listoftup:
                        finsize2d += (tup[0] + _layerclass.kernel_size - 1) * (
                            tup[1] + _layerclass.kernel_size - 1
                        )
                        foutsize2d += tup[0] * tup[1]
                else:
                    finsize2d = _layerclass.feature_size ** 2
                    foutsize2d = (_layerclass.feature_size / _layerclass.stride) ** 2

                fbulk = DataBulk(
                    _modelname,
                    _index + IIleft,
                    datatype,
                    finsize2d * chnum * MoraxConfig.PrecisionBits / 8,
                    fbulkscratch,
                )
                qrf = QueryBuffer(fbulk, BO.Read, ICtypeLeft, CC.TensorCore)
                SubQueryList.append(copy.deepcopy(qrf))
                # make query
                cmos_taskindex += 1
                ctasklabel = make_tasklabel(
                    _modelname, _index, cmos_taskindex, mxLTD[layertype]
                )
                tcbulksize = finsize2d * chnum * MoraxConfig.PrecisionBits / 8
                if bat == 0:
                    tcbulksize += wbulksize
                qe = QueryExcuteOnTC(
                    _layerclass,
                    ctasklabel,
                    "OS",
                    layertype,
                    tasksize_listoftup,
                    tcbulksize,
                )
                SubQueryList.append(copy.deepcopy(qe))
                # make writeback query
                wbbulk = DataBulk(
                    _modelname,
                    _index,
                    "FTR",
                    foutsize2d * MoraxConfig.PrecisionBits / 8,
                    {"B": bat, "HW": fbulkscratch["HW"], "C": comp},
                    token,
                )
                qw = QueryBuffer(wbbulk, BO.Write, CC.TensorCore, CC.FeatureBuffer)
                SubQueryList.append(copy.deepcopy(qw))
        # End for

    if layertype == LLT.TRCONV:
        # for: C B
        ofmapsize = (
            _layerclass.feature_size - 1
        ) * _layerclass.stride * 1.0 + _layerclass.kernel_size
        H = math.ceil(ofmapsize / MoraxConfig.PEArraySize)
        W = H
        C = math.ceil(
            _layerclass.out_channel * 1.0 * H * W / MoraxConfig.PEArrayNum
        )  # ONETC
        H_tail = _layerclass.feature_size % MoraxConfig.PEArraySize
        W_tail = H_tail
        B = _batch
        chnum = 0
        for comp in range(C):
            # spcify task
            tasksize_listoftup = [] * MoraxConfig.PEArrayNum
            this_ch = comp * MoraxConfig.PEArrayNum / (H * W)
            that_ch = (comp + 1) * MoraxConfig.PEArrayNum / (H * W)
            hwidbegin = comp * MoraxConfig.PEArrayNum % (H * W)
            for tmptid in range(MoraxConfig.PEArrayNum):
                tmptup = []
                if this_ch == chsize:
                    tmptup[1] = 0
                    tmptup[2] = 0
                else:
                    ww = hwidbegin % W
                    hh = hwidbegin / W
                    if ww == W - 1 and W_tail > 0:
                        tmptup[1] = W_tail
                    else:
                        tmptup[1] = MoraxConfig.PEArraySize
                    if hh == H - 1 and H_tail > 0:
                        tmptup[0] = H_tail
                    else:
                        tmptup[0] = MoraxConfig.PEArraySize
                tasksize_listoftup[tmptid] = tuple(tmptup)
                hwidbegin += 1
                if (comp * MoraxConfig.PEArrayNum + hwidbegin) / (H * W) > this_ch:
                    assert this_ch < that_ch
                    hwidbegin = 0
                    this_ch += 1
                    chnum += 1
            # make weight op bulk
            wbulkscratch = {}
            datatype = "WET"
            wbulkscratch["K"] = ALL
            wbulkscratch["RS"] = ALL
            wbulkscratch["C"] = comp
            wbulksize = (
                _layerclass.kernel_size ** 2
                * _layerclass.in_channel
                * chnum
                * MoraxConfig.PrecisionBits
                / 8
            )
            wbulk = DataBulk(_modelname, _index, datatype, wbulksize, wbulkscratch,)
            qrw = QueryBuffer(wbulk, BO.Read, CC.WeightBuffer, CC.TensorCore)
            SubQueryList.append(copy.deepcopy(qrw))
            # make feature op bulk
            for bat in range(B):
                fbulkscratch = {}
                datatype = "FTR"
                fbulkscratch["B"] = bat
                fbulkscratch["C"] = ALL
                fbulkscratch["HW"] = (
                    comp * MoraxConfig.PEArrayNum % (H * W) / MoraxConfig.PEArrayNum
                    if H * W >= MoraxConfig.PEArrayNum
                    else ALL
                )
                finsize2d = 0
                foutsize2d = 0
                if H * W >= MoraxConfig.PEArrayNum:
                    for tup in tasksize_listoftup:
                        finsize2d += (tup[0] + _layerclass.kernel_size - 1) * (
                            tup[1] + _layerclass.kernel_size - 1
                        )
                        foutsize2d += tup[0] * tup[1]
                else:
                    finsize2d = _layerclass.feature_size ** 2
                    foutsize2d = (_layerclass.feature_size / _layerclass.stride) ** 2
                # NOTE To avoid misellaneous, Morax use avg infeature num for evaluation
                finsize2d = finsize2d * _layerclass.feature_size / ofmapsize
                fbulk = DataBulk(
                    _modelname,
                    _index + IIleft,
                    datatype,
                    finsize2d * _layerclass.in_channel * MoraxConfig.PrecisionBits / 8,
                    fbulkscratch,
                )
                qrf = QueryBuffer(fbulk, BO.Read, ICtypeLeft, CC.TensorCore)
                SubQueryList.append(copy.deepcopy(qrf))
                # make query
                cmos_taskindex += 1
                ctasklabel = make_tasklabel(
                    _modelname, _index, cmos_taskindex, mxLTD[layertype]
                )
                tcbulksize = (
                    finsize2d * _layerclass.in_channel * MoraxConfig.PrecisionBits / 8
                )
                if bat == 0:
                    tcbulksize += wbulksize
                qe = QueryExcuteOnTC(
                    _layerclass,
                    ctasklabel,
                    "OS",
                    layertype,
                    tasksize_listoftup,
                    tcbulksize,
                )
                SubQueryList.append(copy.deepcopy(qe))
                # make writeback query
                wbbulk = DataBulk(
                    _modelname,
                    _index,
                    "FTR",
                    foutsize2d * MoraxConfig.PrecisionBits / 8,
                    {"B": bat, "HW": fbulkscratch["HW"], "C": comp},
                    token,
                )
                qw = QueryBuffer(wbbulk, BO.Write, CC.TensorCore, CC.FeatureBuffer)
                SubQueryList.append(copy.deepcopy(qw))
        # End for

    if layertype == LLT.NGCONV:
        # for: C B
        H = math.ceil(
            _layerclass.feature_size
            / _layerclass.stride
            * 1.0
            / MoraxConfig.PEArraySize
        )
        W = math.ceil(
            _layerclass.feature_size
            / _layerclass.stride
            * 1.0
            / MoraxConfig.PEArraySize
        )
        C = math.ceil(
            (_layerclass.out_channel / _layerclass.group)
            * 1.0
            * H
            * W
            / MoraxConfig.PEArrayNum
        )  # ONETC
        H_tail = _layerclass.feature_size % MoraxConfig.PEArraySize
        W_tail = H_tail
        B = _batch
        G = _layerclass.group
        for grp in range(G):
            chnum = 0
            for comp in range(C):
                # spcify task
                tasksize_listoftup = [] * MoraxConfig.PEArrayNum
                this_ch = comp * MoraxConfig.PEArrayNum / (H * W)
                that_ch = (comp + 1) * MoraxConfig.PEArrayNum / (H * W)
                hwidbegin = comp * MoraxConfig.PEArrayNum % (H * W)
                for tmptid in range(MoraxConfig.PEArrayNum):
                    tmptup = []
                    if this_ch == chsize:
                        tmptup[1] = 0
                        tmptup[2] = 0
                    else:
                        ww = hwidbegin % W
                        hh = hwidbegin / W
                        if ww == W - 1 and W_tail > 0:
                            tmptup[1] = W_tail
                        else:
                            tmptup[1] = MoraxConfig.PEArraySize
                        if hh == H - 1 and H_tail > 0:
                            tmptup[0] = H_tail
                        else:
                            tmptup[0] = MoraxConfig.PEArraySize
                    tasksize_listoftup[tmptid] = tuple(tmptup)
                    hwidbegin += 1
                    if (comp * MoraxConfig.PEArrayNum + hwidbegin) / (H * W) > this_ch:
                        assert this_ch < that_ch
                        hwidbegin = 0
                        this_ch += 1
                        chnum += 1
                # make weight op bulk
                wbulkscratch = {}
                datatype = "WET"
                wbulkscratch["K"] = grp
                wbulkscratch["RS"] = ALL
                wbulkscratch["C"] = comp
                wbulksize = (
                    _layerclass.kernel_size ** 2
                    * (_layerclass.in_channel / G)
                    * chnum
                    * MoraxConfig.PrecisionBits
                    / 8
                )
                wbulk = DataBulk(_modelname, _index, datatype, wbulksize, wbulkscratch,)
                qrw = QueryBuffer(wbulk, BO.Read, CC.WeightBuffer, CC.TensorCore)
                SubQueryList.append(copy.deepcopy(qrw))
                # make feature op bulk
                for bat in range(B):
                    fbulkscratch = {}
                    datatype = "FTR"
                    fbulkscratch["B"] = bat
                    fbulkscratch["C"] = grp
                    fbulkscratch["HW"] = (
                        comp * MoraxConfig.PEArrayNum % (H * W) / MoraxConfig.PEArrayNum
                        if H * W >= MoraxConfig.PEArrayNum
                        else ALL
                    )
                    finsize2d = 0
                    foutsize2d = 0
                    if H * W >= MoraxConfig.PEArrayNum:
                        for tup in tasksize_listoftup:
                            finsize2d += (tup[0] + _layerclass.kernel_size - 1) * (
                                tup[1] + _layerclass.kernel_size - 1
                            )
                            foutsize2d += tup[0] * tup[1]
                    else:
                        finsize2d = _layerclass.feature_size ** 2
                        foutsize2d = (
                            _layerclass.feature_size / _layerclass.stride
                        ) ** 2
                    fbulk = DataBulk(
                        _modelname,
                        _index + IIleft,
                        datatype,
                        finsize2d
                        * (_layerclass.in_channel / G)
                        * MoraxConfig.PrecisionBits
                        / 8,
                        fbulkscratch,
                    )
                    qrf = QueryBuffer(fbulk, BO.Read, ICtypeLeft, CC.TensorCore)
                    SubQueryList.append(copy.deepcopy(qrf))
                    # make query
                    cmos_taskindex += 1
                    ctasklabel = make_tasklabel(
                        _modelname, _index, cmos_taskindex, mxLTD[layertype]
                    )
                    tcbulksize = (
                        finsize2d
                        * (_layerclass.in_channel / G)
                        * MoraxConfig.PrecisionBits
                        / 8
                    )
                    if bat == 0:
                        tcbulksize += wbulksize
                    qe = QueryExcuteOnTC(
                        _layerclass,
                        ctasklabel,
                        "OS",
                        layertype,
                        tasksize_listoftup,
                        tcbulksize,
                    )
                    SubQueryList.append(copy.deepcopy(qe))
                    # make writeback query
                    wbbulk = DataBulk(
                        _modelname,
                        _index,
                        "FTR",
                        foutsize2d * MoraxConfig.PrecisionBits / 8,
                        {"B": bat, "HW": fbulkscratch["HW"], "C": grp * C + comp},
                        token,
                    )
                    qw = QueryBuffer(wbbulk, BO.Write, CC.TensorCore, CC.FeatureBuffer)
                    SubQueryList.append(copy.deepcopy(qw))
        # End for

    if layertype == LLT.Residual or layertype == LLT.MADD:
        # for: C B OR B C
        hsize = (
            _layerclass.feature_size
            if layertype == LLT.Residual
            else _layerclass.row_dim
        )
        wsize = (
            _layerclass.feature_size
            if layertype == LLT.Residual
            else _layerclass.col_dim
        )
        chsize = _layerclass.channel if layertype == LLT.Residual else 1
        H = math.ceil(hsize * 1.0 / MoraxConfig.PEArraySize)
        W = math.ceil(wsize * 1.0 / MoraxConfig.PEArraySize)
        C = math.ceil(chsize * 1.0 * H * W / MoraxConfig.PEArrayNum)  # ONETC
        H_tail = hsize % MoraxConfig.PEArraySize
        W_tail = wsize % MoraxConfig.PEArraySize
        B = _batch
        # TODO: CLEAN UP THIS
        chnum = 0
        for comp in range(C):
            # spcify task
            tasksize_listoftup = [] * MoraxConfig.PEArrayNum
            this_ch = comp * MoraxConfig.PEArrayNum / (H * W)
            that_ch = (comp + 1) * MoraxConfig.PEArrayNum / (H * W)
            hwidbegin = comp * MoraxConfig.PEArrayNum % (H * W)
            for tmptid in range(MoraxConfig.PEArrayNum):
                tmptup = []
                if this_ch == chsize:
                    tmptup[1] = 0
                    tmptup[2] = 0
                else:
                    ww = hwidbegin % W
                    hh = hwidbegin / W
                    if ww == W - 1 and W_tail > 0:
                        tmptup[1] = W_tail
                    else:
                        tmptup[1] = MoraxConfig.PEArraySize
                    if hh == H - 1 and H_tail > 0:
                        tmptup[0] = H_tail
                    else:
                        tmptup[0] = MoraxConfig.PEArraySize
                tasksize_listoftup[tmptid] = tuple(tmptup)
                hwidbegin += 1
                if (comp * MoraxConfig.PEArrayNum + hwidbegin) / (H * W) > this_ch:
                    assert this_ch < that_ch
                    hwidbegin = 0
                    this_ch += 1
                    chnum += 1
            # make left right op bulk
            for bat in range(B):
                mbulkscratch = {}
                fbulkscratch = {}
                if layertype == LLT.Residual:
                    datatype = "FTR"
                    fbulkscratch["B"] = bat
                    fbulkscratch["C"] = comp
                    fbulkscratch["HW"] = (
                        comp * MoraxConfig.PEArrayNum % (H * W) / MoraxConfig.PEArrayNum
                        if H * W >= MoraxConfig.PEArrayNum
                        else ALL
                    )
                    fsize2d = 0
                    if H * W >= MoraxConfig.PEArrayNum:
                        for tup in tasksize_listoftup:
                            fsize2d += tup[0] * tup[1]
                    else:
                        fsize2d = _layerclass.feature_size ** 2 * chnum
                    bulkleft = DataBulk(
                        _modelname,
                        _index + IIleft,
                        datatype,
                        fsize2d * MoraxConfig.PrecisionBits / 8,
                        fbulkscratch,
                    )
                    bulkright = DataBulk(
                        _modelname,
                        _index + IIright,
                        datatype,
                        fsize2d * MoraxConfig.PrecisionBits / 8,
                        fbulkscratch,
                    )
                else:
                    datatype = "MAT"
                    mbulkscratch["B"] = bat
                    mbulkscratch["N"] = (
                        comp * MoraxConfig.PEArrayNum % (H * W) / MoraxConfig.PEArrayNum
                        if H * W >= MoraxConfig.PEArrayNum
                        else ALL
                    )
                    mbulkscratch["M"] = mbulkscratch["N"]
                    msize2d = 0
                    for tup in tasksize_listoftup:
                        msize2d += tup[0] * tup[1]
                    bulkleft = DataBulk(
                        _modelname,
                        _index + IIleft,
                        datatype,
                        msize2d * MoraxConfig.PrecisionBits / 8,
                        mbulkscratch,
                    )
                    bulkright = DataBulk(
                        _modelname,
                        _index + IIright,
                        datatype,
                        msize2d * MoraxConfig.PrecisionBits / 8,
                        mbulkscratch,
                    )
                qrl = QueryBuffer(bulkleft, BO.Read, ICtypeLeft, CC.TensorCore)
                SubQueryList.append(copy.deepcopy(qrl))
                qrr = QueryBuffer(bulkright, BO.Read, ICtypeRight, CC.TensorCore)
                SubQueryList.append(copy.deepcopy(qrr))
                # make query
                cmos_taskindex += 1
                ctasklabel = make_tasklabel(
                    _modelname, _index, cmos_taskindex, mxLTD[layertype]
                )
                if layertype == LLT.Residual:
                    tcbulksize = fsize2d * 2 * MoraxConfig.PrecisionBits / 8
                else:
                    tcbulksize = msize2d * 2 * MoraxConfig.PrecisionBits / 8
                qe = QueryExcuteOnTC(
                    _layerclass,
                    ctasklabel,
                    "OS",
                    layertype,
                    tasksize_listoftup,
                    tcbulksize,
                )
                SubQueryList.append(copy.deepcopy(qe))
                # make writeback
                if layertype == LLT.Residual:
                    wbbulk = DataBulk(
                        _modelname,
                        _index,
                        "FTR",
                        fsize2d * MoraxConfig.PrecisionBits / 8,
                        {"B": bat, "HW": fbulkscratch["HW"], "C": comp},
                        token,
                    )
                else:
                    wbbulk = DataBulk(
                        _modelname,
                        _index,
                        "MAT",
                        msize2d * MoraxConfig.PrecisionBits / 8,
                        {"B": bat, "M": mbulkscratch["M"], "N": mbulkscratch["N"]},
                        token,
                    )
                qw = QueryBuffer(wbbulk, BO.Write, CC.TensorCore, CC.FeatureBuffer)
                SubQueryList.append(copy.deepcopy(qw))
        # End for

    if layertype == LLT.Batchnorm:
        # for: C B
        H = math.ceil(_layerclass.feature_size * 1.0 / MoraxConfig.PEArraySize)
        W = math.ceil(_layerclass.feature_size * 1.0 / MoraxConfig.PEArraySize)
        C = math.ceil(
            _layerclass.channel * 1.0 * H * W / MoraxConfig.PEArrayNum
        )  # ONETC
        H_tail = _layerclass.feature_size % MoraxConfig.PEArraySize
        W_tail = H_tail
        B = _batch
        chnum = 0
        for comp in range(C):
            # spcify task
            tasksize_listoftup = [] * MoraxConfig.PEArrayNum
            this_ch = comp * MoraxConfig.PEArrayNum / (H * W)
            that_ch = (comp + 1) * MoraxConfig.PEArrayNum / (H * W)
            hwidbegin = comp * MoraxConfig.PEArrayNum % (H * W)
            for tmptid in range(MoraxConfig.PEArrayNum):
                tmptup = []
                if this_ch == chsize:
                    tmptup[1] = 0
                    tmptup[2] = 0
                else:
                    ww = hwidbegin % W
                    hh = hwidbegin / W
                    if ww == W - 1 and W_tail > 0:
                        tmptup[1] = W_tail
                    else:
                        tmptup[1] = MoraxConfig.PEArraySize
                    if hh == H - 1 and H_tail > 0:
                        tmptup[0] = H_tail
                    else:
                        tmptup[0] = MoraxConfig.PEArraySize
                tasksize_listoftup[tmptid] = tuple(tmptup)
                hwidbegin += 1
                if (comp * MoraxConfig.PEArrayNum + hwidbegin) / (H * W) > this_ch:
                    assert this_ch < that_ch
                    hwidbegin = 0
                    this_ch += 1
                    chnum += 1
            # make lut query
            lut_taskindex = -1
            for chn in range(chnum):
                lut_taskindex += 1
                luttasklabel = make_tasklabel(
                    _modelname, _index, lut_taskindex, mxLTD[layertype]
                )
                lutadress = get_lookup_adress(_modelname, _index, this_ch + chn)
                qlut = QueryExcuteOnNVTC(
                    _layerclass,
                    luttasklabel,
                    "LUT16",
                    SO.LookUp,
                    lutadress[0],
                    lutadress[1],
                    lutadress[2],
                )
                SubQueryList.append(copy.deepcopy(qlut))
            # make feature op bulk
            for bat in range(B):
                fbulkscratch = {}
                datatype = "FTR"
                fbulkscratch["B"] = bat
                fbulkscratch["C"] = comp
                fbulkscratch["HW"] = (
                    comp * MoraxConfig.PEArrayNum % (H * W) / MoraxConfig.PEArrayNum
                    if H * W >= MoraxConfig.PEArrayNum
                    else ALL
                )
                fsize2d = 0
                if H * W >= MoraxConfig.PEArrayNum:
                    for tup in tasksize_listoftup:
                        fsize2d += tup[0] * tup[1]
                else:
                    fsize2d = _layerclass.feature_size ** 2 * chnum
                fbulk = DataBulk(
                    _modelname,
                    _index + IIleft,
                    datatype,
                    fsize2d * MoraxConfig.PrecisionBits / 8,
                    fbulkscratch,
                )
                qrf = QueryBuffer(fbulk, BO.Read, ICtypeLeft, CC.TensorCore)
                SubQueryList.append(copy.deepcopy(qrf))
                # make query
                cmos_taskindex += 1
                ctasklabel = make_tasklabel(
                    _modelname, _index, cmos_taskindex, mxLTD[layertype]
                )
                tcbulksize = fsize2d * MoraxConfig.PrecisionBits / 8
                qe = QueryExcuteOnTC(
                    _layerclass,
                    ctasklabel,
                    "OS",
                    layertype,
                    tasksize_listoftup,
                    tcbulksize,
                )
                SubQueryList.append(copy.deepcopy(qe))
                # make writeback
                wbbulk = DataBulk(
                    _modelname,
                    _index,
                    "FTR",
                    fsize2d * MoraxConfig.PrecisionBits / 8,
                    {"B": bat, "HW": fbulkscratch["HW"], "C": comp},
                    token,
                )
                qw = QueryBuffer(wbbulk, BO.Write, CC.TensorCore, CC.FeatureBuffer)
                SubQueryList.append(copy.deepcopy(qw))
        # End for

    if layertype == LLT.Layernorm:
        # for: C B
        M = math.ceil(_layerclass.row_dim * 1.0 / MoraxConfig.PEArraySize)
        N = math.ceil(_layerclass.col_dim * 1.0 / MoraxConfig.PEArraySize)
        M_tail = _layerclass.row_dim % MoraxConfig.PEArraySize
        N_tail = _layerclass.col_dim % MoraxConfig.PEArraySize
        B = math.ceil(_batch * 1.0 * M * N / MoraxConfig.PEArrayNum)
        chnum = 0
        for bomp in range(B):
            # spcify task
            tasksize_listoftup = [] * MoraxConfig.PEArrayNum
            this_ch = bomp * MoraxConfig.PEArrayNum / (M * N)
            that_ch = (bomp + 1) * MoraxConfig.PEArrayNum / (M * N)
            mnidbegin = bomp * MoraxConfig.PEArrayNum % (M * N)
            for tmptid in range(MoraxConfig.PEArrayNum):
                tmptup = []
                if this_ch == chsize:
                    tmptup[1] = 0
                    tmptup[2] = 0
                else:
                    nn = mnidbegin % N
                    mm = mnidbegin / N
                    if mm == M - 1 and M_tail > 0:
                        tmptup[0] = M_tail
                    else:
                        tmptup[0] = MoraxConfig.PEArraySize
                    if nn == N - 1 and N_tail > 0:
                        tmptup[1] = N_tail
                    else:
                        tmptup[1] = MoraxConfig.PEArraySize
                tasksize_listoftup[tmptid] = tuple(tmptup)
                mnidbegin += 1
                if (bomp * MoraxConfig.PEArrayNum + mnidbegin) / (M * N) > this_ch:
                    assert this_ch < that_ch
                    mnidbegin = 0
                    this_ch += 1
                    chnum += 1
            # make lut query
            lut_taskindex = -1
            for chn in range(chnum):
                lut_taskindex += 1
                luttasklabel = make_tasklabel(
                    _modelname, _index, lut_taskindex, mxLTD[layertype]
                )
                lutadress = get_lookup_adress(_modelname, _index, this_ch + chn)
                qlut = QueryExcuteOnNVTC(
                    _layerclass,
                    luttasklabel,
                    "LUT16",
                    SO.LookUp,
                    lutadress[0],
                    lutadress[1],
                    lutadress[2],
                )
                SubQueryList.append(copy.deepcopy(qlut))
            # make feature op bulk
            mbulkscratch = {}
            datatype = "MAT"
            mbulkscratch["B"] = bomp
            mbulkscratch["N"] = (
                bomp * MoraxConfig.PEArrayNum % (N * M) / MoraxConfig.PEArrayNum
                if H * W >= MoraxConfig.PEArrayNum
                else ALL
            )
            mbulkscratch["M"] = mbulkscratch["N"]
            fsize2d = 0
            if H * W >= MoraxConfig.PEArrayNum:
                for tup in tasksize_listoftup:
                    fsize2d += tup[0] * tup[1]
            else:
                fsize2d = _layerclass.feature_size ** 2 * chnum
            mbulk = DataBulk(
                _modelname,
                _index + IIleft,
                datatype,
                fsize2d * MoraxConfig.PrecisionBits / 8,
                mbulkscratch,
            )
            qrm = QueryBuffer(mbulk, BO.Read, ICtypeLeft, CC.TensorCore)
            SubQueryList.append(copy.deepcopy(qrm))
            # make query
            cmos_taskindex += 1
            ctasklabel = make_tasklabel(
                _modelname, _index, cmos_taskindex, mxLTD[layertype]
            )
            tcbulksize = fsize2d * MoraxConfig.PrecisionBits / 8
            qe = QueryExcuteOnTC(
                _layerclass,
                ctasklabel,
                "OS",
                layertype,
                tasksize_listoftup,
                tcbulksize,
            )
            SubQueryList.append(copy.deepcopy(qe))
            # make writeback
            wbbulk = DataBulk(
                _modelname,
                _index,
                "MAT",
                fsize2d * MoraxConfig.PrecisionBits / 8,
                {"B": bomp, "M": mbulkscratch["HW"], "N": mbulkscratch["HW"]},
                token,
            )
            qw = QueryBuffer(wbbulk, BO.Write, CC.TensorCore, CC.FeatureBuffer)
            SubQueryList.append(copy.deepcopy(qw))
        # End for

    if layertype == LLT.GEMM:
        if IIleft == 0:
            row_dim = _layerclass.m_dim
            col_dim = _layerclass.k_dim
            v_dim = _layerclass.n_dim
        else:
            row_dim = _layerclass.k_dim
            col_dim = _layerclass.n_dim
            v_dim = _layerclass.m_dim
        M = math.ceil(row_dim * 1.0 / MoraxConfig.PEArraySize)
        N = math.ceil(col_dim * 1.0 / MoraxConfig.PEArraySize)
        M_tail = row_dim % MoraxConfig.RRAMXbarSize
        N_tail = col_dim % MoraxConfig.RRAMXbarSize
        B = _batch
        # for m in range(M):
        # BMN
        pepart = 0
        listof_tasksize_listoftup = []
        listof_minfo_list = []
        listof_ninfo_list = []
        while pepart < M * N:
            tasksize_listoftup = []
            minfo_list = []
            ninfo_list = []
            for peid in range(MoraxConfig.PEArrayNum):
                pesizetup = []
                if pepart + peid >= M * N:
                    pesizetup[0] = 0
                    pesizetup[1] = 0
                else:
                    mm = (pepart + peid) / N
                    nn = (pepart + peid) % N
                    if mm == M - 1 and M_tail > 0:
                        pesizetup[0] = M_tail
                    else:
                        pesizetup[0] = MoraxConfig.PEArraySize
                    if nn == N - 1 and N_tail > 0:
                        pesizetup[1] = N_tail
                    else:
                        pesizetup[1] = MoraxConfig.PEArraySize
                    if mm not in minfo_list:
                        minfo_list.append(mm)
                    if nn not in ninfo_list:
                        ninfo_list.append(nn)
                tasksize_listoftup.append(tuple(pesizetup))
            listof_tasksize_listoftup.append(tasksize_listoftup)
            listof_minfo_list.append(minfo_list)
            listof_ninfo_list.append(ninfo_list)
            pepart += MoraxConfig.PEArrayNum
        LOTL = listof_tasksize_listoftup
        LOML = listof_minfo_list
        LONL = listof_ninfo_list
        for tctask_idx in range(len(LOTL)):
            minfo_list = LOML[tctask_idx]
            tasksize_listoftup = LOTL[tctask_idx]
            # make weight op bulk
            wmbulkscratch = {}
            datatype = "MAT"
            wmbulkscratch["B"] = ALL
            wmbulkscratch["M"] = minfo_list
            wmbulkscratch["N"] = ALL
            wmbulksize = (
                len(ninfo_list)
                * MoraxConfig.PEArraySize
                * row_dim
                * MoraxConfig.PrecisionBits
                / 8
            )
            if IIleft == 0:
                wmbulk = DataBulk(
                    _modelname, _index + IIleft, datatype, wmbulksize, wmbulkscratch,
                )
                qrwm = QueryBuffer(wmbulk, BO.Read, ICtypeLeft, CC.TensorCore)
                SubQueryList.append(copy.deepcopy(qrwm))
            elif IIright == 0:
                wmbulk = DataBulk(
                    _modelname, _index + IIright, datatype, wmbulksize, wmbulkscratch,
                )
                qrwm = QueryBuffer(wmbulk, BO.Read, ICtypeRight, CC.TensorCore)
                SubQueryList.append(copy.deepcopy(qrwm))
            # make input op bulk
            for bat in range(B):
                imbulkscratch = {}
                datatype = "MAT"
                imbulkscratch["B"] = bat
                imbulkscratch["M"] = minfo_list
                imbulkscratch["N"] = ALL
                imbulksize = (
                    len(minfo_list)
                    * MoraxConfig.PEArraySize
                    * row_dim
                    * MoraxConfig.PrecisionBits
                    / 8
                )
                if IIleft < 0:
                    imbulk = DataBulk(
                        _modelname,
                        _index + IIleft,
                        datatype,
                        imbulksize,
                        imbulkscratch,
                    )
                    qrim = QueryBuffer(imbulk, BO.Read, ICtypeLeft, CC.TensorCore)
                    SubQueryList.append(copy.deepcopy(qrwm))
                if IIright < 0:
                    imbulk = DataBulk(
                        _modelname,
                        _index + IIright,
                        datatype,
                        imbulksize,
                        imbulkscratch,
                    )
                    qrim = QueryBuffer(imbulk, BO.Read, ICtypeRight, CC.TensorCore)
                    SubQueryList.append(copy.deepcopy(qrim))
                # make query
                cmos_taskindex += 1
                ctasklabel = make_tasklabel(
                    _modelname, _index, cmos_taskindex, mxLTD[layertype]
                )
                tcbulksize = imbulksize
                if bat == 0:
                    tcbulksize += wmbulksize
                qe = QueryExcuteOnTC(
                    _layerclass,
                    ctasklabel,
                    "OS",
                    layertype,
                    tasksize_listoftup,
                    tcbulksize,
                )
                SubQueryList.append(copy.deepcopy(qe))
                # make writeback query
                msize2d = 0
                for tup in tasksize_listoftup:
                    msize2d += tup[0] * tup[1]
                wbbulk = DataBulk(
                    _modelname,
                    _index,
                    "MAT",
                    msize2d * MoraxConfig.PrecisionBits / 8,
                    {"B": bat, "M": minfo_list, "N": ninfo_list},
                    token,
                )
                qw = QueryBuffer(wbbulk, BO.Write, CC.TensorCore, CC.FeatureBuffer)
                SubQueryList.append(copy.deepcopy(qw))
        # End for
    # End CMOS
    return SubQueryList


def compileVPU(_index, _modelname, _layerclass, _batch, _token):
    SubQueryList = []
    layertype = _layerclass.layer_type
    vpu_taskindex = -1
    smu_taskindex = -1
    lut_taskindex = -1
    # Now only consider NLT
    if layertype == NLT.Pooling:
        B = _batch
        C = _layerclass.channel
        ofsize = _layerclass.feature_size / _layerclass.kernel_size
        for bat in range(B):
            for ch in range(C):
                # rf
                datatype = "FTR"
                fbulkscratch = {}
                fbulkscratch["B"] = bat
                fbulkscratch["HW"] = ALL
                fbulkscratch["C"] = ch
                fbs = _layerclass.feature_size ** 2 * MoraxConfig.PrecisionBits / 8
                fbulk = DataBulk(
                    _modelname,
                    _index + _layerclass.input_indecies_tuple[0],
                    datatype,
                    fbs,
                    fbulkscratch,
                )
                qrf = QueryBuffer(fbulk, BO.Read, CC.FeatureBuffer, CC.VPU)
                SubQueryList.append(copy.deepcopy(qrf))
                # query
                vpu_taskindex += 1
                vtasklabel = make_tasklabel(
                    _modelname, _index, vpu_taskindex, mxNTD[layertype]
                )
                qv = QueryExcuteOnVPU(_layerclass, vtasklabel, "Linear", NLT.Pooling,)
                SubQueryList.append(copy.deepcopy(qv))
                # wb
                wbs = ofsize ** 2 * MoraxConfig.PrecisionBits / 8
                wbbulk = DataBulk(_modelname, _index, datatype, wbs, fbulkscratch,)
                qwb = QueryBuffer(wbbulk, BO.Write, CC.VPU, CC.FeatureBuffer)
                SubQueryList.append(copy.deepcopy(qwb))
        # End for

    if layertype == NLT.Softmax1D or NLT.Softmax2D:
        B = _batch
        L = 1 if layertype == NLT.Softmax1D else _layerclass.row_dim
        V = _layerclass.v_dim if layertype == NLT.Softmax1D else _layerclass.col_dim
        for bat in range(B):
            for line in range(L):
                # read
                datatype = "VEC" if layertype == NLT.Softmax1D else "MAT"
                rbulkscratch = {}
                rbulkscratch["B"] = bat
                rbulkscratch["M"] = line if layertype == NLT.Softmax2D else ALL
                rbulkscratch["N"] = ALL
                rbs = V * MoraxConfig.PrecisionBits / 8
                rbulk = DataBulk(
                    _modelname,
                    _index + _layerclass.input_indecies_tuple[0],
                    datatype,
                    rbs,
                    rbulkscratch,
                )
                qr = QueryBuffer(rbulk, BO.Read, CC.FeatureBuffer, CC.VPU)
                SubQueryList.append(copy.deepcopy(qr))
                # vmax
                vpu_taskindex += 1
                vtasklabel = make_tasklabel(
                    _modelname, _index, vpu_taskindex, mxNTD[layertype]
                )
                qv = QueryExcuteOnVPU(_layerclass, vtasklabel, "Softmax", SO.VMAX,)
                SubQueryList.append(copy.deepcopy(qv))
                # SO.Truncation
                smu_taskindex += 1
                stasklabel = make_tasklabel(
                    _modelname, _index, smu_taskindex, mxNTD[layertype]
                )
                qsmu = QueryExcuteOnSMU(
                    _layerclass, stasklabel, "UpStream", SO.Truncation,
                )
                SubQueryList.append(copy.deepcopy(qsmu))
                # exp lut
                for v in range(V):
                    lut_taskindex += 1
                    luttasklabel = make_tasklabel(
                        _modelname, _index, lut_taskindex, mxLTD[layertype]
                    )
                    lutadress = get_lookup_adress(_modelname, _index, line)
                    qlut = QueryExcuteOnNVTC(
                        _layerclass,
                        luttasklabel,
                        "LUT8",
                        SO.LookUp,
                        lutadress[0],
                        lutadress[1],
                        lutadress[2],
                    )
                    SubQueryList.append(copy.deepcopy(qlut))
                # norm
                # lutdiv
                lut_taskindex += 1
                luttasklabel = make_tasklabel(
                    _modelname, _index, lut_taskindex, mxLTD[layertype]
                )
                lutadress = get_lookup_adress(_modelname, _index, line)
                qlut = QueryExcuteOnNVTC(
                    _layerclass,
                    luttasklabel,
                    "LUT16",
                    SO.LookUp,
                    lutadress[0],
                    lutadress[1],
                    lutadress[2],
                )
                SubQueryList.append(copy.deepcopy(qlut))
                # vnrom
                vpu_taskindex += 1
                vtasklabel = make_tasklabel(
                    _modelname, _index, vpu_taskindex, mxNTD[layertype]
                )
                qv = QueryExcuteOnVPU(_layerclass, vtasklabel, "Softmax", SO.VNORM,)
                SubQueryList.append(copy.deepcopy(qv))
                # write
                wbulk = DataBulk(_modelname, _index, datatype, rbs, rbulkscratch,)
                qw = QueryBuffer(wbulk, BO.Write, CC.VPU, CC.FeatureBuffer)
                SubQueryList.append(copy.deepcopy(qw))
        # Eno for

    return SubQueryList


def make_tc_task_list(_tasklen):
    ttltup = []
    for peid in range(MoraxConfig.PEArrayNum):
        if _tasklen >= MoraxConfig.PEArraySize:
            _tasklen -= MoraxConfig.PEArraySize
            ttltup[peid] = (MoraxConfig.PEArraySize, MoraxConfig.PEArraySize)
        elif _tasklen > 0:
            _tasklen -= MoraxConfig.PEArraySize
            ttltup[peid] = (MoraxConfig.PEArraySize, _tasklen)
        else:
            ttltup[peid] = (0, 0)
    return ttltup
