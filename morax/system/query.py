# system query class
# Morax means the mixing of CMOS and RRAM
# Author: Tao Song.
# Last modification: 0316


from ast import Assign
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
import math
from morax.model.model import ModelDAG, ModelList, ModelType
from morax.frontend.api import (
    get_idx_from_concat,
    get_lookup_adress,
)


# [bulk]
# indicate the data form of input and weight
# bulkfrom = (part or whole) feature: NCHW  weight: KCRS  MVM & GEMM:
# dataname = W or F or
# bulklabel = modelname_'L'+layeridx_'dataname'+bulkidx_bulksizeByte_bulkfrom

# [task]
# indicate the task excution using [output]
# taskform = (part or whole) batchN outputchannelC heighH widthW  MVM: outputdimO batchN GEMM: heightH widthW batchN
# CONV
# RRAM: CHWN onefeaturebyonefeature    OS: HWNC onechannelbyonechannel (FOR MAX KERNEL REUSE)
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


class VritualQuerySeparator(VritualQuery):
    def __init__(self):
        super().__init__(VO.QuerySeprator)


class QueryBuffer:
    def __init__(self, _databulkclass: DataBulk, _execution, _locationEnum, _toEnum):
        self.execution = _execution
        self.databulkclass = copy.deepcopy(_databulkclass)
        self.locationEnum = _locationEnum
        self.toEnum = _toEnum
        self.clusterid = -1

    def update_clusterid(self, _id):
        self.clusterid = _id

    def location_assigned(self):
        return self.clusterid >= 0


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
        _tasksizelist: list,
        # list of tuple (row, col)
        _bulksize,  # add 03.28
    ):
        super().__init__(_layerclass, _tasklabel)
        self.dfmod = _dfmod
        self.execution = _execution
        self.tasksizelist = copy.deepcopy(_tasksizelist)
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
        self.sliceidlist = copy.deepcopy(_sliceidlist)
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
        _tasksize: tuple = (0, 0),
    ):
        super().__init__(_layerclass, _tasklabel)
        self.dfmod = _dfmod
        self.execution = _execution
        self.tasksize = copy.deepcopy(_tasksize)  # (rowparts, collines)
        # assert self.checkquery() is True

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
        _tasksize: tuple = (0, 0),
    ):
        super().__init__(_layerclass, _tasklabel)
        self.dfmod = _dfmod
        self.execution = _execution
        self.tasksize = copy.deepcopy(_tasksize)
        assert self.checkquery() is True

    def checkquery(self):
        dfsafe = self.dfmod == "UpStream" or self.dfmod == "DownStream"
        executionsafe = self.execution in MoraxExecutionDict[ClusterComponent.SMU]
        tasksafe = True
        return dfsafe and executionsafe and tasksafe


class QueryRingBus:
    def __init__(
        self,
        _databulkclass: DataBulk,
        _subbulksize: int,
        _fromCluster,
        _toCluster,
        _worf,
    ):
        self.databulkclass = copy.deepcopy(_databulkclass)
        self.subbulksizebyte = _subbulksize
        self.fromCluster = _fromCluster
        self.toCluster = _toCluster
        self.worf = _worf


class QueryDMA:
    def __init__(self, _databulkclass: DataBulk, _toCluster, _worf):
        self.databulkclass = _databulkclass
        self.toCluster = _toCluster
        self.worf = _worf


""" ============================================================================================================
                     generate layer query and compilie them to sub query
================================================================================================================
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
        self.assignment = copy.deepcopy(_assignment)
        self.layerclass = copy.deepcopy(_layerclass)
        self.iodegree = {"in": _indegree, "out": _outdegree}
        self.subquerynum = 0
        self.SubQueryList = []
        self.ISSUE_TIME = -1
        self.SUBMIT_TIME = -1
        self.FINISHED_FLAG = False

        # add info to monitor
        # self.outscratch = get_layer_scratchdict(_layerclass)
        # self.outdatatype = get_datatype(_layerclass)

        # assignment:
        # listoftuple_clstid_nvtcid_sliceidlist
        # default []
        """
        if isinstance(self.assignment, list):
            assert mapper.check_mapping_with_query(
                self.layerclass.layer_index
            ), "{}".format(self.layerclass.layer_name)
        """

    def set_issue_t(self, _issue_t):
        assert _issue_t > 0
        self.ISSUE_TIME = _issue_t

    def set_submit_t(self, _submit_t):
        assert _submit_t > 0
        self.SUBMIT_TIME = _submit_t
        self.FINISHED_FLAG = True

    def get_submit_t(self):
        assert self.FINISHED_FLAG
        return self.SUBMIT_TIME

    def compile(self, _modelname, _moraxchip, _concatlist=[]):
        # Generate subqueries of this layer query
        # print("[Morax][System] Compiling Query {}.".format(self.q_index))
        layertype = self.layerclass.layer_type
        if isinstance(layertype, LLT):
            if layertype == LLT.CONCAT:
                self.SubQueryList = copy.deepcopy(
                    compileCONCAT(self.q_index, self.layerclass, _concatlist)
                )
            else:
                if len(self.assignment) > 0:  # on NVTC
                    self.SubQueryList = copy.deepcopy(
                        compileRRAM(
                            self.q_index,
                            _modelname,
                            self.layerclass,
                            self.assignment,
                            _moraxchip,
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
                            self.batch,
                            self.iodegree["out"],
                        )
                    )
        elif isinstance(layertype, NLT):
            self.SubQueryList = copy.deepcopy(
                compileVPU(
                    self.q_index,
                    _modelname,
                    self.layerclass,
                    self.batch,
                    self.iodegree["out"],
                )
            )
        self.subquerynum = len(self.SubQueryList)
        return


def generate_demmy_queries(
    _modelDAG: ModelDAG, _batch=1,
):
    print("[Morax][System] Compile demmy queries.")
    chip = 0
    totalquery = 0
    for idx in _modelDAG.LayerIndexList:
        q = LayerQuery(
            idx,
            _batch,
            _modelDAG.LayerClassDict[idx],
            {},
            len(_modelDAG.fromVertexDict[idx]),
            len(_modelDAG.toVertexDict[idx]),
        )  # TODO: CHANGE INDEX TUPLE of Layerclass
        q.compile(_modelDAG.modelname, chip, _modelDAG.ConcatList)
        _modelDAG.LayerQueryClassDict[idx] = copy.deepcopy(q)
        totalquery += 1
    assert totalquery == _modelDAG.layernum


def generate_queries(
    _modelDAG: ModelDAG, _moraxchip, _batch=1,
):
    print("[Morax][System] Compile queries.")
    totalquery = 0
    for idx in _modelDAG.LayerIndexList:
        q = LayerQuery(
            idx,
            _batch,
            _modelDAG.LayerClassDict[idx],
            _modelDAG.LayerAssignmentDict[idx],
            len(_modelDAG.fromVertexDict[idx]),
            len(_modelDAG.toVertexDict[idx]),
        )  # TODO: CHANGE INDEX TUPLE of Layerclass
        print("[Morax][System] Compiling query {}".format(idx))
        q.compile(_modelDAG.modelname, _moraxchip, _modelDAG.ConcatList)
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
    _index, _modelname, _layerclass, _doclotnsl: dict, _chip, _batch, token
):
    # NOTE BATCH IS ALWAYS THE LAST
    # dict of clstid: [tuple1(nvtcid, sliceidlist), tuple2, ... ]
    SubQueryList = []
    layertype = _layerclass.layer_type
    (IIleft, IIright) = _layerclass.input_indecies_tuple
    xbar_size = [
        MoraxConfig.RRAMXbarSize - MoraxConfig.RRAMLUTRows,
        MoraxConfig.RRAMXbarSize,
    ]
    if layertype == LLT.Linear or layertype == LLT.VMM:
        # (CLSTNVTC B) V WB
        M = math.ceil(_layerclass.row_dim * 1.0 / xbar_size[0])
        N = math.ceil(_layerclass.col_dim * 1.0 / xbar_size[1])
        M_tail = _layerclass.row_dim % xbar_size[0]
        N_tail = _layerclass.col_dim % xbar_size[1]
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
                            .layerinfo[2]
                        )
                    bulkinfo = []
                    for tup in mapinfo:
                        if tup[0] not in bulkinfo:
                            bulkinfo.append(tup[0])
                    # make query
                    datatype = (
                        "FTR"
                        if (layertype == LLT.Linear and _layerclass.is_fc)
                        else "VEC"
                    )
                    bulkscratch = {}
                    taskindex += 1
                    bulkscratch["B"] = bat
                    if datatype == "FTR":
                        bulkscratch["H"] = 0
                        bulkscratch["W"] = 0
                        rowpart = "C"
                    else:
                        rowpart = "M"
                    bsize = 0
                    fst, lst = bulkinfo[0], bulkinfo[0]
                    for inf in bulkinfo:
                        if inf == M and M_tail > 0:
                            bsize += M_tail
                        else:
                            bsize += xbar_size[0]
                        fst = inf if inf < fst else fst
                        lst = inf if lst < inf else lst
                    last = (
                        (lst + 1) * xbar_size[0] - 1
                        if lst != M
                        else _layerclass.row_dim
                    )
                    bulkscratch[rowpart] = (
                        fst * xbar_size[0],
                        last,
                    )
                    bsize = bsize * MoraxConfig.PrecisionBits // 8
                    # for bat in range(_batch):
                    bulk = DataBulk(
                        _modelname, _index + +IIleft, datatype, bsize, bulkscratch
                    )
                    qr = QueryBuffer(bulk, BO.Read, CC.FeatureBuffer, CC.nvTensorCore)
                    tasklabel = make_tasklabel(_modelname, _index, taskindex, "Linear")
                    qr.update_clusterid(clstid)  # add 0521
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
                    # SubQueryList.append(VritualQuerySeparator())
            tasklabel = make_tasklabel(_modelname, _index, bat, "PostProcess")
            qv = QueryExcuteOnVPU(
                _layerclass,
                tasklabel,
                "PostProcess",
                LLT.Linear,
                (M, _layerclass.col_dim),
            )
            SubQueryList.append(copy.deepcopy(qv))
            bulk = DataBulk(
                _modelname,
                _index,
                "VEC",
                _layerclass.col_dim * MoraxConfig.PrecisionBits // 8,
                {"B": bat, "M": (0, _layerclass.col_dim - 1)},
                token,
            )
            qw = QueryBuffer(bulk, BO.Write, CC.VPU, CC.FeatureBuffer)
            SubQueryList.append(copy.deepcopy(qw))
            SubQueryList.append(VritualQuerySeparator())
        # End for

    if layertype == LLT.CONV:
        M = math.ceil(
            _layerclass.in_channel * _layerclass.kernel_size ** 2 * 1.0 / xbar_size[0]
        )
        N = math.ceil(_layerclass.out_channel * 1.0 / xbar_size[1])
        M_tail = _layerclass.in_channel * _layerclass.kernel_size ** 2 % xbar_size[0]
        N_tail = _layerclass.out_channel % xbar_size[1]
        omapsize = _layerclass.feature_size // _layerclass.stride
        rram_taskindex = -1
        vpu_taskindex = -1
        for bat in range(_batch):
            for row_iter in range(omapsize):
                for col_iter in range(omapsize):
                    # make simple bulk
                    datatype = "FTR"
                    bulkscratch = {}
                    bulkscratch["B"] = bat
                    hbegin = 0 if row_iter == 0 else (row_iter - 1) * _layerclass.stride
                    hend = (
                        _layerclass.feature_size - 1
                        if row_iter == omapsize - 1
                        else (row_iter - 1) * _layerclass.stride
                        + _layerclass.kernel_size
                    )
                    wbegin = 0 if col_iter == 0 else (col_iter - 1) * _layerclass.stride
                    wend = (
                        _layerclass.feature_size - 1
                        if col_iter == omapsize - 1
                        else (col_iter - 1) * _layerclass.stride
                        + _layerclass.kernel_size
                    )
                    bulkscratch["W"] = (hbegin, hend)
                    bulkscratch["H"] = (wbegin, wend)
                    if IIright != 0:
                        bulkscratch["C"] = (0, _layerclass.in_channel // 2 - 1)
                    else:
                        bulkscratch["C"] = (0, _layerclass.in_channel - 1)
                    bsize = (
                        _layerclass.kernel_size
                        * _layerclass.in_channel
                        * MoraxConfig.PrecisionBits
                        // 8
                    )
                    bsize *= (
                        _layerclass.kernel_size if col_iter == 0 else _layerclass.stride
                    )
                    bulk = DataBulk(
                        _modelname, _index + IIleft, datatype, bsize, bulkscratch
                    )
                    qr = QueryBuffer(bulk, BO.Read, CC.FeatureBuffer, CC.nvTensorCore)
                    SubQueryList.append(copy.deepcopy(qr))
                    if IIright != 0:
                        bulk2 = DataBulk(
                            _modelname, _index + IIright, datatype, bsize, bulkscratch
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
                    wbscratch = {}
                    wbscratch["B"] = bat
                    wbscratch["H"] = row_iter
                    wbscratch["W"] = col_iter
                    wbscratch["C"] = (0, _layerclass.out_channel)
                    wbbulk = DataBulk(
                        _modelname,
                        _index,
                        "FTR",
                        _layerclass.out_channel * MoraxConfig.PrecisionBits // 8,
                        wbscratch,
                        token,
                    )
                    qw = QueryBuffer(wbbulk, BO.Write, CC.VPU, CC.FeatureBuffer)
                    SubQueryList.append(copy.deepcopy(qw))
                    SubQueryList.append(VritualQuerySeparator())
            # End for

    if layertype == LLT.TRCONV:
        M = math.ceil(
            _layerclass.in_channel * _layerclass.kernel_size ** 2 * 1.0 / xbar_size[0]
        )
        N = math.ceil(_layerclass.out_channel * 1.0 / xbar_size[1])
        M_tail = _layerclass.in_channel * _layerclass.kernel_size ** 2 % xbar_size[0]
        N_tail = _layerclass.out_channel % xbar_size[1]
        omapsize = (
            _layerclass.feature_size - 1
        ) * _layerclass.stride + _layerclass.kernel_size
        rram_taskindex = -1
        vpu_taskindex = -1
        for bat in range(_batch):
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
                        # make simple bulk is have new data in window
                        datatype = "FTR"
                        bulkscratch = {}
                        bulkscratch["B"] = bat
                        bulkscratch["W"] = (
                            col_iter // _layerclass.stride - 1,
                            col_iter // _layerclass.stride - 1,
                        )
                        bulkscratch["H"] = (
                            row_iter // _layerclass.stride - 1,
                            row_iter // _layerclass.stride - 1,
                        )
                        bulkscratch["C"] = (0, _layerclass.in_channel - 1)
                        bsize = _layerclass.in_channel * MoraxConfig.PrecisionBits // 8
                        bulk = DataBulk(
                            _modelname, _index + IIleft, datatype, bsize, bulkscratch
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
                    wbscratch = {}
                    wbscratch["B"] = bat
                    wbscratch["H"] = row_iter
                    wbscratch["W"] = col_iter
                    wbscratch["C"] = (0, _layerclass.out_channel)
                    wbbulk = DataBulk(
                        _modelname,
                        _index,
                        "FTR",
                        _layerclass.out_channel * MoraxConfig.PrecisionBits // 8,
                        wbscratch,
                        token,
                    )
                    qw = QueryBuffer(wbbulk, BO.Write, CC.VPU, CC.FeatureBuffer)
                    SubQueryList.append(copy.deepcopy(qw))
                    SubQueryList.append(VritualQuerySeparator())
            # End for

    if layertype == LLT.NGCONV:
        M = math.ceil(
            (_layerclass.in_channel / _layerclass.group)
            * _layerclass.kernel_size ** 2
            * 1.0
            / xbar_size[0]
        )
        N = math.ceil(
            (_layerclass.out_channel / _layerclass.group) * 1.0 / xbar_size[1]
        )
        M_tail = (
            (_layerclass.in_channel // _layerclass.group)
            * _layerclass.kernel_size ** 2
            / xbar_size[0]
        )
        N_tail = (_layerclass.out_channel // _layerclass.group) / xbar_size[1]
        rram_taskindex = -1
        vpu_taskindex = -1
        group_inchannel = _layerclass.in_channel // _layerclass.group
        group_outchannel = _layerclass.out_channel // _layerclass.group
        omapsize = _layerclass.feature_size // _layerclass.stride
        for bat in range(_batch):
            for grp in range(_layerclass.group):
                for row_iter in range(omapsize):
                    for col_iter in range(omapsize):
                        # make simple bulk
                        datatype = "FTR"
                        bulkscratch = {}
                        bulkscratch["B"] = bat
                        hbegin = (
                            0 if row_iter == 0 else (row_iter - 1) * _layerclass.stride
                        )
                        hend = (
                            _layerclass.feature_size - 1
                            if row_iter == omapsize - 1
                            else (row_iter - 1) * _layerclass.stride
                            + _layerclass.kernel_size
                        )
                        wbegin = (
                            0 if col_iter == 0 else (col_iter - 1) * _layerclass.stride
                        )
                        wend = (
                            _layerclass.feature_size - 1
                            if col_iter == omapsize - 1
                            else (col_iter - 1) * _layerclass.stride
                            + _layerclass.kernel_size
                        )
                        bulkscratch["W"] = (hbegin, hend)
                        bulkscratch["H"] = (wbegin, wend)
                        bulkscratch["C"] = (
                            grp * group_inchannel,
                            (grp + 1) * group_inchannel - 1,
                        )
                        bsize = (
                            _layerclass.kernel_size
                            * group_inchannel
                            * (MoraxConfig.PrecisionBits // 8)
                        )
                        bsize *= (
                            _layerclass.kernel_size
                            if col_iter == 0
                            else _layerclass.stride
                        )
                        bulk = DataBulk(
                            _modelname, _index + IIleft, datatype, bsize, bulkscratch
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
                            (M, group_outchannel),
                        )
                        SubQueryList.append(copy.deepcopy(qv))
                        # make writeback bulk
                        wbscratch = {}
                        wbscratch["B"] = bat
                        wbscratch["H"] = row_iter
                        wbscratch["W"] = col_iter
                        wbscratch["C"] = (
                            grp * group_outchannel,
                            (grp + 1) * group_outchannel - 1,
                        )
                        wbbulk = DataBulk(
                            _modelname,
                            _index,
                            "FTR",
                            group_outchannel * (MoraxConfig.PrecisionBits // 8),
                            wbscratch,
                            token,
                        )
                        qw = QueryBuffer(bulk, BO.Write, CC.VPU, CC.FeatureBuffer)
                        SubQueryList.append(copy.deepcopy(qw))
                        SubQueryList.append(VritualQuerySeparator())
        # End for

    if layertype == LLT.GEMM:
        """
              =========       =========
        mdim  =========   *   =========   kdim
              =========       =========
                              =========
                              ndim
        RRRR EEEE
        """
        # idxtup = _layerclass.input_indecies_tuple
        """
        if IIright == 0:
            iidx = IIleft
            m_dim = _layerclass.k_dim
            k_dim = _layerclass.n_dim
            n_dim = _layerclass.m_dim
        else:
            iidx = IIright
            m_dim = _layerclass.m_dim
            k_dim = _layerclass.k_dim
            n_dim = _layerclass.n_dim
        """  # 0605
        if IIleft == 0:
            iidx = IIright
            m_dim = _layerclass.n_dim
            k_dim = _layerclass.k_dim
            n_dim = _layerclass.m_dim
        else:
            iidx = IIleft
            m_dim = _layerclass.m_dim
            k_dim = _layerclass.k_dim
            n_dim = _layerclass.n_dim
        M = math.ceil(k_dim * 1.0 / xbar_size[0])
        N = math.ceil(n_dim * 1.0 / xbar_size[1])
        M_tail = k_dim % xbar_size[0]
        N_tail = n_dim % xbar_size[1]
        # assume fetch batch is 16 = NVTCNUM/2
        vfetch = 16
        V = math.ceil(m_dim * 1.0 / vfetch)
        rram_taskindex = -1
        vpu_taskindex = -1
        for bat in range(_batch):
            for vf in range(V):
                # match Model info to Chip info
                for clstid, nvtclist in _doclotnsl.items():
                    for nvtctup in nvtclist:
                        (nvtcid, sliceidlist) = nvtctup
                        mapinfo = []
                        for info in sliceidlist:
                            mapinfo.append(
                                _chip.ClusterList[clstid]
                                .nvTensorCoreList[nvtcid]
                                .RRAMSliceObjList[info]
                                .layerinfo[2]
                            )
                        bulkinfo = []
                        for tup in mapinfo:
                            if tup[0] not in bulkinfo:
                                bulkinfo.append(tup[0])
                        # make input matrix query
                        datatype = "MAT"
                        bulkscratch = {}
                        bulkscratch["B"] = bat
                        bulkscratch["M"] = (vf * vfetch, (vf + 1) * vfetch - 1)
                        # bulkscratch["N"] = []
                        bsize = 0
                        fst, lst = bulkinfo[0], bulkinfo[0]
                        for inf in bulkinfo:
                            if inf == M and M_tail > 0:
                                bsize += M_tail
                            else:
                                bsize += xbar_size[0]
                            fst = inf if inf < fst else fst
                            lst = inf if lst < inf else lst
                        last = (lst + 1) * xbar_size[0] - 1 if lst != M else k_dim
                        bulkscratch["N"] = (
                            fst * xbar_size[0],
                            last,
                        )
                        bsize = vfetch * bsize * MoraxConfig.PrecisionBits // 8
                        bulk = DataBulk(
                            _modelname, _index + iidx, datatype, bsize, bulkscratch
                        )
                        qr = QueryBuffer(
                            bulk, BO.Read, CC.FeatureBuffer, CC.nvTensorCore
                        )
                        qr.update_clusterid(clstid)  # add 0521
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
                        _modelname, _index, vpu_taskindex, "PostProcess"
                    )
                    qv = QueryExcuteOnVPU(
                        _layerclass, vtasklabel, "PostProcess", LLT.GEMM, (M, n_dim),
                    )
                    SubQueryList.append(copy.deepcopy(qv))
                    wbbulk = DataBulk(
                        _modelname,
                        _index,
                        "MAT",
                        n_dim * MoraxConfig.PrecisionBits // 8,
                        {"B": bat, "M": line + vf * vfetch, "N": (0, n_dim - 1)},
                        token,
                    )
                    qw = QueryBuffer(wbbulk, BO.Write, CC.VPU, CC.FeatureBuffer)
                    SubQueryList.append(copy.deepcopy(qw))
                    SubQueryList.append(VritualQuerySeparator())
            # End for
    return SubQueryList


def compileCMOS(_index, _modelname, _layerclass, _batch, token):
    # NOTE BATCH IS ALWAYS assigned on one same tc
    SubQueryList = []
    layertype = _layerclass.layer_type
    onetcsize = MoraxConfig.PEArraySize * MoraxConfig.PEArrayNum
    (IIleft, IIright) = _layerclass.input_indecies_tuple
    # TODO check dag topo again
    if IIleft == 0 and IIright < 0:
        ICtypeLeft = CC.WeightBuffer
        ICtypeRight = CC.FeatureBuffer
    elif IIleft < 0 and IIright == 0:
        ICtypeLeft = CC.FeatureBuffer
        ICtypeRight = CC.WeightBuffer
    elif IIleft < 0 and IIright < 0:
        ICtypeLeft = CC.FeatureBuffer
        ICtypeRight = CC.FeatureBuffer
    else:
        ICtypeLeft = CC.FeatureBuffer
        ICtypeRight = CC.WeightBuffer
    cmos_taskindex = -1
    vpu_taskindex = -1
    # go
    if layertype == LLT.Linear or layertype == LLT.VMM:
        # for: M N B
        # (R RRR EEEE)
        M = math.ceil(_layerclass.row_dim * 1.0 / MoraxConfig.PEArraySize)
        N = math.ceil(_layerclass.col_dim * 1.0 / onetcsize)
        N_tail = _layerclass.col_dim % onetcsize
        B = _batch
        for m in range(M):
            # make left op bulk
            fbulkscratch = {}
            datatype = (
                "FTR" if (layertype == LLT.Linear and _layerclass.is_fc) else "VEC"
            )
            fbulkscratch["B"] = (0, B - 1)
            if datatype == "FTR":
                fbulkscratch["W"] = 0
                fbulkscratch["H"] = 0
                fbulkscratch["C"] = (
                    m * MoraxConfig.PEArraySize,
                    (m + 1) * MoraxConfig.PEArraySize - 1,
                )
            else:
                fbulkscratch["M"] = (
                    m * MoraxConfig.PEArraySize,
                    (m + 1) * MoraxConfig.PEArraySize - 1,
                )
            fbulk = DataBulk(
                _modelname,
                _index + IIleft,
                datatype,
                B * MoraxConfig.PEArraySize * MoraxConfig.PrecisionBits // 8,
                fbulkscratch,
            )
            qrf = QueryBuffer(fbulk, BO.Read, ICtypeLeft, CC.TensorCore)
            SubQueryList.append(copy.deepcopy(qrf))
            # make right op bulk, on tc size
            for n in range(N):
                wbulkscratch = {}
                datatype = "MAT"
                wbulkscratch["B"] = 0
                wbulkscratch["M"] = (
                    m * MoraxConfig.PEArraySize,
                    (m + 1) * MoraxConfig.PEArraySize - 1,
                )
                wbulkscratch["N"] = (n * onetcsize, (n + 1) * onetcsize - 1)
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
                        _modelname, _index, cmos_taskindex, "Linear"
                    )
                    if n == N - 1 and N_tail > 0:
                        tasksize_list = copy.deepcopy(make_tc_task_list(N_tail))
                    else:
                        tasksize_list = [
                            (MoraxConfig.PEArraySize, MoraxConfig.PEArraySize)
                        ] * MoraxConfig.PEArrayNum
                    tcbulksize = (
                        MoraxConfig.PEArraySize * MoraxConfig.PrecisionBits // 8
                    )
                    tskbulksize = 0
                    for s in range(MoraxConfig.PEArrayNum):
                        tskbulksize += tasksize_list[s][0] * tasksize_list[s][1]
                    if b == 0:
                        tcbulksize = (
                            tcbulksize
                            + tskbulksize
                            * MoraxConfig.PEArraySize
                            * MoraxConfig.PrecisionBits
                            // 8
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
                SubQueryList.append(VritualQuerySeparator())
        # make vpu query after all tc exe
        for b in range(B):
            vbulk = DataBulk(_modelname, _index, "VEC", 0, {"B": b, "M": 0},)
            SubQueryList.append(
                copy.deepcopy(QueryBuffer(vbulk, BO.Read, CC.FeatureBuffer, CC.VPU))
            )
            vtasklabel = make_tasklabel(_modelname, _index, b, "PostProcess")
            qv = QueryExcuteOnVPU(
                _layerclass,
                vtasklabel,
                "PostProcess",
                layertype,
                (M, _layerclass.col_dim),
            )
            SubQueryList.append(copy.deepcopy(qv))
            wbbulk = DataBulk(
                _modelname,
                _index,
                "VEC",
                _layerclass.col_dim * MoraxConfig.PrecisionBits // 8,
                {"B": b, "M": (0, _layerclass.col_dim - 1)},
                token,
            )
            qw = QueryBuffer(wbbulk, BO.Write, CC.VPU, CC.FeatureBuffer)
            SubQueryList.append(copy.deepcopy(qw))
        SubQueryList.append(VritualQuerySeparator())

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
        TSK = math.ceil(
            _layerclass.out_channel * 1.0 * H * W / MoraxConfig.PEArrayNum
        )  # TC num
        P = math.ceil(_layerclass.out_channel * 1.0 * H * W)
        H_tail = _layerclass.feature_size % MoraxConfig.PEArraySize
        W_tail = H_tail
        B = _batch
        omapsize = _layerclass.feature_size // _layerclass.stride
        # Spcify os task
        assigned_pearray = 0
        listof_tasksize_listoftup = []
        listof_scratchdict = []
        while assigned_pearray < P:  # 1 - PE
            tasksize_listoftup = []
            [cb, ce] = [0, 0]
            [hb, he] = [0, 0]
            [wb, we] = [0, 0]
            for peid in range(MoraxConfig.PEArrayNum):  # 0 - num-1
                pe_tasksize_tup = [0, 0]
                if assigned_pearray + peid >= P:  # last outofrange pes
                    pe_tasksize_tup[0] = 0
                    pe_tasksize_tup[1] = 0
                else:
                    cc = (assigned_pearray + peid) // (H * W)
                    hh = (assigned_pearray + peid) % (H * W) // W
                    ww = (assigned_pearray + peid) % (H * W) % W
                    if peid == 0:
                        [cb, ce] = [cc, cc]
                        [hb, he] = [hh, hh]
                        [wb, we] = [ww, ww]
                    else:
                        ce = cc if cc > cb else ce
                        he = hh
                        we = ww
                    if hh == H - 1 and H_tail > 0:
                        pe_tasksize_tup[0] = H_tail
                    else:
                        pe_tasksize_tup[0] = MoraxConfig.PEArraySize
                    if ww == W - 1 and W_tail > 0:
                        pe_tasksize_tup[1] = W_tail
                    else:
                        pe_tasksize_tup[1] = MoraxConfig.PEArraySize
                tasksize_listoftup.append(pe_tasksize_tup)
            hbegin = hb * MoraxConfig.PEArraySize
            hend = (
                omapsize - 1 if he == H - 1 else (he + 1) * MoraxConfig.PEArraySize - 1
            )
            wbegin = wb * MoraxConfig.PEArraySize
            wend = (
                omapsize - 1 if we == W - 1 else (we + 1) * MoraxConfig.PEArraySize - 1
            )
            bulkscratch = {}
            # bulkscratch["B"] = ALL
            bulkscratch["C"] = (cb, ce)
            bulkscratch["H"] = (hbegin, hend)
            bulkscratch["W"] = (wbegin, wend)
            listof_tasksize_listoftup.append(copy.deepcopy(tasksize_listoftup))
            listof_scratchdict.append(copy.deepcopy(bulkscratch))
            assigned_pearray += MoraxConfig.PEArrayNum
        # End spcify
        # Make Bulk
        # (R (REW))
        for taskid in range(TSK):
            tasksize_listoftup = listof_tasksize_listoftup[taskid]
            ofbulkscratch = listof_scratchdict[taskid]
            chnum = len(ofbulkscratch["C"])
            # Make Weight Bulk
            wbulkscratch = {}
            wbulkscratch["K"] = ofbulkscratch["C"]
            wbulkscratch["RS"] = 0
            wbulkscratch["C"] = (0, _layerclass.in_channel - 1)
            wbulksize = (
                _layerclass.kernel_size ** 2
                * _layerclass.in_channel
                * chnum
                * MoraxConfig.PrecisionBits
                // 8
            )
            wbulk = DataBulk(_modelname, _index, "WET", wbulksize, wbulkscratch,)
            qrw = QueryBuffer(wbulk, BO.Read, CC.WeightBuffer, CC.TensorCore)
            SubQueryList.append(copy.deepcopy(qrw))
            # Make Feature Bulk
            finsize = 0
            fosize = 0
            for pe_tasksize_tup in tasksize_listoftup:
                finsize += (
                    pe_tasksize_tup[0] * _layerclass.stride
                    + _layerclass.kernel_size
                    - 1
                ) * (
                    pe_tasksize_tup[1] * _layerclass.stride
                    + _layerclass.kernel_size
                    - 1
                )
                fosize += pe_tasksize_tup[0] * pe_tasksize_tup[1]
            inch = (
                _layerclass.in_channel if IIright == 0 else _layerclass.in_channel // 2
            )
            ifbulkscratch = {}
            ifbulkscratch["C"] = (0, inch - 1)
            hbegin = (
                0
                if ofbulkscratch["H"][0] == 0
                else ofbulkscratch["H"][0] * _layerclass.stride - 1
            )
            hend = (
                _layerclass.feature_size - 1
                if ofbulkscratch["H"][1] == omapsize - 1
                else (
                    ofbulkscratch["H"][1] * _layerclass.stride
                    - 1
                    + _layerclass.kernel_size
                    - 1
                )
            )
            wbegin = (
                0
                if ofbulkscratch["W"][0] == 0
                else ofbulkscratch["W"][0] * _layerclass.stride - 1
            )
            wend = (
                _layerclass.feature_size - 1
                if ofbulkscratch["W"][1] == omapsize - 1
                else (
                    ofbulkscratch["W"][1] * _layerclass.stride
                    - 1
                    + _layerclass.kernel_size
                    - 1
                )
            )
            ifbulkscratch["H"] = (hbegin, hend)
            ifbulkscratch["W"] = (wbegin, wend)
            # todo check it
            for bat in range(B):
                ifbulkscratch["B"] = bat
                ofbulkscratch["B"] = bat
                ifbulk = DataBulk(
                    _modelname,
                    _index + IIleft,
                    "FTR",
                    finsize * inch * MoraxConfig.PrecisionBits // 8,
                    ifbulkscratch,
                )
                qrf = QueryBuffer(ifbulk, BO.Read, ICtypeLeft, CC.TensorCore)
                SubQueryList.append(copy.deepcopy(qrf))
                if IIright != 0:
                    ifbulk2 = DataBulk(
                        _modelname,
                        _index + IIright,
                        "FTR",
                        finsize * inch * MoraxConfig.PrecisionBits // 8,
                        ifbulkscratch,
                    )
                    qrf2 = QueryBuffer(ifbulk2, BO.Read, ICtypeRight, CC.TensorCore)
                    SubQueryList.append(copy.deepcopy(qrf2))
                # Make query
                cmos_taskindex += 1
                tctasklabel = make_tasklabel(_modelname, _index, cmos_taskindex, "CONV")
                tcbulksize = (
                    finsize * _layerclass.in_channel * MoraxConfig.PrecisionBits // 8
                )
                if bat == 0:
                    tcbulksize += wbulksize
                qe = QueryExcuteOnTC(
                    _layerclass,
                    tctasklabel,
                    "OS",
                    layertype,
                    tasksize_listoftup,
                    tcbulksize,
                )
                SubQueryList.append(copy.deepcopy(qe))
                # Make writeback query
                wbbulk = DataBulk(
                    _modelname,
                    _index,
                    "FTR",
                    fosize * MoraxConfig.PrecisionBits // 8,
                    ofbulkscratch,
                    token,
                )
                qw = QueryBuffer(wbbulk, BO.Write, CC.TensorCore, CC.FeatureBuffer)
                SubQueryList.append(copy.deepcopy(qw))
            SubQueryList.append(VritualQuerySeparator())
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
        TSK = math.ceil(
            _layerclass.channel * 1.0 * H * W / MoraxConfig.PEArrayNum
        )  # TC num
        P = math.ceil(_layerclass.channel * 1.0 * H * W)
        H_tail = _layerclass.feature_size % MoraxConfig.PEArraySize
        W_tail = H_tail
        B = _batch
        omapsize = _layerclass.feature_size // _layerclass.stride
        # Spcify os task
        assigned_pearray = 0
        listof_tasksize_listoftup = []
        listof_scratchdict = []
        while assigned_pearray < P:  # 1 - PE
            tasksize_listoftup = []
            [cb, ce] = [0, 0]
            [hb, he] = [0, 0]
            [wb, we] = [0, 0]
            for peid in range(MoraxConfig.PEArrayNum):  # 0 - num-1
                pe_tasksize_tup = [0, 0]
                if assigned_pearray + peid >= P:  # last outofrange pes
                    pe_tasksize_tup[0] = 0
                    pe_tasksize_tup[1] = 0
                else:
                    cc = (assigned_pearray + peid) // (H * W)
                    hh = (assigned_pearray + peid) % (H * W) // W
                    ww = (assigned_pearray + peid) % (H * W) % W
                    if peid == 0:
                        [cb, ce] = [cc, cc]
                        [hb, he] = [hh, hh]
                        [wb, we] = [ww, ww]
                    else:
                        ce = cc if cc > cb else ce
                        he = hh
                        we = ww
                    if hh == H - 1 and H_tail > 0:
                        pe_tasksize_tup[0] = H_tail
                    else:
                        pe_tasksize_tup[0] = MoraxConfig.PEArraySize
                    if ww == W - 1 and W_tail > 0:
                        pe_tasksize_tup[1] = W_tail
                    else:
                        pe_tasksize_tup[1] = MoraxConfig.PEArraySize
                tasksize_listoftup.append(pe_tasksize_tup)
            hbegin = hb * MoraxConfig.PEArraySize
            hend = (
                omapsize - 1 if he == H - 1 else (he + 1) * MoraxConfig.PEArraySize - 1
            )
            wbegin = wb * MoraxConfig.PEArraySize
            wend = (
                omapsize - 1 if we == W - 1 else (we + 1) * MoraxConfig.PEArraySize - 1
            )
            bulkscratch = {}
            # bulkscratch["B"] = ALL
            bulkscratch["C"] = (cb, ce)
            bulkscratch["H"] = (hbegin, hend)
            bulkscratch["W"] = (wbegin, wend)
            listof_tasksize_listoftup.append(copy.deepcopy(tasksize_listoftup))
            listof_scratchdict.append(copy.deepcopy(bulkscratch))
            assigned_pearray += MoraxConfig.PEArrayNum
        # End spcify
        # Make Bulk
        for taskid in range(TSK):
            tasksize_listoftup = listof_tasksize_listoftup[taskid]
            ofbulkscratch = listof_scratchdict[taskid]
            chnum = len(ofbulkscratch["C"])
            # Make Weight Bulk
            wbulkscratch = {}
            wbulkscratch["K"] = ofbulkscratch["C"]
            wbulkscratch["RS"] = 0
            wbulkscratch["C"] = 0
            wbulksize = (
                _layerclass.kernel_size ** 2 * chnum * MoraxConfig.PrecisionBits // 8
            )
            wbulk = DataBulk(_modelname, _index, "WET", wbulksize, wbulkscratch,)
            qrw = QueryBuffer(wbulk, BO.Read, CC.WeightBuffer, CC.TensorCore)
            SubQueryList.append(copy.deepcopy(qrw))
            # Make Feature Bulk
            finsize = 0
            fosize = 0
            for pe_tasksize_tup in tasksize_listoftup:
                finsize += (
                    pe_tasksize_tup[0] * _layerclass.stride
                    + _layerclass.kernel_size
                    - 1
                ) * (
                    pe_tasksize_tup[1] * _layerclass.stride
                    + _layerclass.kernel_size
                    - 1
                )
                fosize += pe_tasksize_tup[0] * pe_tasksize_tup[1]
            ifbulkscratch = {}
            ifbulkscratch["C"] = ofbulkscratch["C"]
            hbegin = (
                0
                if ofbulkscratch["H"][0] == 0
                else ofbulkscratch["H"][0] * _layerclass.stride - 1
            )
            hend = (
                _layerclass.feature_size - 1
                if ofbulkscratch["H"][1] == omapsize - 1
                else (
                    ofbulkscratch["H"][1] * _layerclass.stride
                    - 1
                    + _layerclass.kernel_size
                    - 1
                )
            )
            wbegin = (
                0
                if ofbulkscratch["W"][0] == 0
                else ofbulkscratch["W"][0] * _layerclass.stride - 1
            )
            wend = (
                _layerclass.feature_size - 1
                if ofbulkscratch["W"][1] == omapsize - 1
                else (
                    ofbulkscratch["W"][1] * _layerclass.stride
                    - 1
                    + _layerclass.kernel_size
                    - 1
                )
            )
            ifbulkscratch["H"] = (hbegin, hend)
            ifbulkscratch["W"] = (wbegin, wend)
            for bat in range(B):
                ifbulkscratch["B"] = bat
                ofbulkscratch["B"] = bat
                ifbulk = DataBulk(
                    _modelname,
                    _index + IIleft,
                    "FTR",
                    finsize * MoraxConfig.PrecisionBits // 8,
                    ifbulkscratch,
                )
                qrf = QueryBuffer(ifbulk, BO.Read, ICtypeLeft, CC.TensorCore)
                SubQueryList.append(copy.deepcopy(qrf))
                # Make query
                cmos_taskindex += 1
                tctasklabel = make_tasklabel(
                    _modelname, _index, cmos_taskindex, "DWCONV"
                )
                tcbulksize = finsize * MoraxConfig.PrecisionBits // 8
                if bat == 0:
                    tcbulksize += wbulksize
                qe = QueryExcuteOnTC(
                    _layerclass,
                    tctasklabel,
                    "OS",
                    layertype,
                    tasksize_listoftup,
                    tcbulksize,
                )
                SubQueryList.append(copy.deepcopy(qe))
                # Make writeback query
                wbbulk = DataBulk(
                    _modelname,
                    _index,
                    "FTR",
                    fosize * MoraxConfig.PrecisionBits // 8,
                    ofbulkscratch,
                    token,
                )
                qw = QueryBuffer(wbbulk, BO.Write, CC.TensorCore, CC.FeatureBuffer)
                SubQueryList.append(copy.deepcopy(qw))
            SubQueryList.append(VritualQuerySeparator())
        # End for

    if layertype == LLT.TRCONV:
        # for: C B
        omapsize = (
            _layerclass.feature_size - 1
        ) * _layerclass.stride * 1.0 + _layerclass.kernel_size
        H = math.ceil(omapsize / MoraxConfig.PEArraySize)
        W = H
        TSK = math.ceil(_layerclass.out_channel * 1.0 * H * W / MoraxConfig.PEArrayNum)
        P = math.ceil(_layerclass.channel * 1.0 * H * W)
        H_tail = _layerclass.feature_size % MoraxConfig.PEArraySize
        W_tail = H_tail
        B = _batch
        # Spcify os task
        assigned_pearray = 0
        listof_tasksize_listoftup = []
        listof_scratchdict = []
        while assigned_pearray < P:  # 1 - PE
            tasksize_listoftup = []
            [cb, ce] = [0, 0]
            [hb, he] = [0, 0]
            [wb, we] = [0, 0]
            for peid in range(MoraxConfig.PEArrayNum):  # 0 - num-1
                pe_tasksize_tup = [0, 0]
                if assigned_pearray + peid >= P:  # last outofrange pes
                    pe_tasksize_tup[0] = 0
                    pe_tasksize_tup[1] = 0
                else:
                    cc = (assigned_pearray + peid) // (H * W)
                    hh = (assigned_pearray + peid) % (H * W) // W
                    ww = (assigned_pearray + peid) % (H * W) % W
                    if peid == 0:
                        [cb, ce] = [cc, cc]
                        [hb, he] = [hh, hh]
                        [wb, we] = [ww, ww]
                    else:
                        ce = cc if cc > cb else ce
                        he = hh
                        we = ww
                    if hh == H - 1 and H_tail > 0:
                        pe_tasksize_tup[0] = H_tail
                    else:
                        pe_tasksize_tup[0] = MoraxConfig.PEArraySize
                    if ww == W - 1 and W_tail > 0:
                        pe_tasksize_tup[1] = W_tail
                    else:
                        pe_tasksize_tup[1] = MoraxConfig.PEArraySize
                tasksize_listoftup.append(pe_tasksize_tup)
            hbegin = hb * MoraxConfig.PEArraySize
            hend = (
                omapsize - 1 if he == H - 1 else (he + 1) * MoraxConfig.PEArraySize - 1
            )
            wbegin = wb * MoraxConfig.PEArraySize
            wend = (
                omapsize - 1 if we == W - 1 else (we + 1) * MoraxConfig.PEArraySize - 1
            )
            bulkscratch = {}
            # bulkscratch["B"] = ALL
            bulkscratch["C"] = (cb, ce)
            bulkscratch["H"] = (hbegin, hend)
            bulkscratch["W"] = (wbegin, wend)
            listof_tasksize_listoftup.append(copy.deepcopy(tasksize_listoftup))
            listof_scratchdict.append(copy.deepcopy(bulkscratch))
            assigned_pearray += MoraxConfig.PEArrayNum
        # End spcify
        # Make Bulk
        for taskid in range(TSK):
            tasksize_listoftup = listof_tasksize_listoftup[taskid]
            ofbulkscratch = listof_scratchdict[taskid]
            chnum = len(ofbulkscratch["C"])
            # Make Weight Bulk
            wbulkscratch = {}
            wbulkscratch["K"] = ofbulkscratch["C"]
            wbulkscratch["RS"] = 0
            wbulkscratch["C"] = (0, _layerclass.in_channel - 1)
            wbulksize = (
                _layerclass.kernel_size ** 2
                * _layerclass.in_channel
                * chnum
                * MoraxConfig.PrecisionBits
                // 8
            )
            wbulk = DataBulk(_modelname, _index, "WET", wbulksize, wbulkscratch,)
            qrw = QueryBuffer(wbulk, BO.Read, CC.WeightBuffer, CC.TensorCore)
            SubQueryList.append(copy.deepcopy(qrw))
            # Make Feature Bulk
            finsize = 0
            fosize = 0
            for pe_tasksize_tup in tasksize_listoftup:
                finsize += (
                    pe_tasksize_tup[0] * _layerclass.stride
                    + _layerclass.kernel_size
                    - 1
                ) * (
                    pe_tasksize_tup[1] * _layerclass.stride
                    + _layerclass.kernel_size
                    - 1
                )
                fosize += pe_tasksize_tup[0] * pe_tasksize_tup[1]
            ifbulkscratch = {}
            ifbulkscratch["C"] = (0, _layerclass.in_channel - 1)
            # NOTE To avoid misellaneous, Morax use avg infeature num for evaluation
            hbegin = (
                taskid * _layerclass.feature_size ** 2 // TSK
            ) / _layerclass.feature_size
            hend = (
                (taskid + 1) * _layerclass.feature_size ** 2 // TSK
            ) / _layerclass.feature_size
            wbegin = (
                taskid * _layerclass.feature_size ** 2 // TSK
            ) % _layerclass.feature_size
            wend = (
                (taskid + 1) * _layerclass.feature_size ** 2 // TSK
            ) % _layerclass.feature_size
            ifbulkscratch["H"] = (
                (hbegin, hend)
                if hend < _layerclass.feature_size
                else (hbegin, _layerclass.feature_size - 1)
            )
            ifbulkscratch["W"] = (
                (wbegin, wend)
                if wend < _layerclass.feature_size
                else (wbegin, _layerclass.feature_size - 1)
            )
            # todo check it
            for bat in range(B):
                ifbulkscratch["B"] = bat
                ofbulkscratch["B"] = bat
                ifbulk = DataBulk(
                    _modelname,
                    _index + IIleft,
                    "FTR",
                    finsize * _layerclass.in_channel * MoraxConfig.PrecisionBits // 8,
                    ifbulkscratch,
                )
                qrf = QueryBuffer(ifbulk, BO.Read, ICtypeLeft, CC.TensorCore)
                SubQueryList.append(copy.deepcopy(qrf))
                # Make query
                cmos_taskindex += 1
                tctasklabel = make_tasklabel(
                    _modelname, _index, cmos_taskindex, "TRCONV"
                )
                tcbulksize = (
                    finsize * _layerclass.in_channel * MoraxConfig.PrecisionBits // 8
                )
                if bat == 0:
                    tcbulksize += wbulksize
                qe = QueryExcuteOnTC(
                    _layerclass,
                    tctasklabel,
                    "OS",
                    layertype,
                    tasksize_listoftup,
                    tcbulksize,
                )
                SubQueryList.append(copy.deepcopy(qe))
                # Make writeback query
                wbbulk = DataBulk(
                    _modelname,
                    _index,
                    "FTR",
                    fosize * MoraxConfig.PrecisionBits // 8,
                    ofbulkscratch,
                    token,
                )
                qw = QueryBuffer(wbbulk, BO.Write, CC.TensorCore, CC.FeatureBuffer)
                SubQueryList.append(copy.deepcopy(qw))
            SubQueryList.append(VritualQuerySeparator())
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
        TSK = math.ceil(
            (_layerclass.out_channel / _layerclass.group)
            * 1.0
            * H
            * W
            / MoraxConfig.PEArrayNum
        )
        P = math.ceil(_layerclass.out_channel * 1.0 * H * W / _layerclass.group)
        H_tail = _layerclass.feature_size % MoraxConfig.PEArraySize
        W_tail = H_tail
        B = _batch
        G = _layerclass.group
        omapsize = _layerclass.feature_size // _layerclass.stride
        group_inc = _layerclass.in_channel // _layerclass.group
        group_outc = _layerclass.out_channel // _layerclass.group
        for grp in range(G):
            # Spcify OS task
            assigned_pearray = 0
            listof_tasksize_listoftup = []
            listof_scratchdict = []
            while assigned_pearray < P:  # 1 - PE
                tasksize_listoftup = []
                [cb, ce] = [0, 0]
                [hb, he] = [0, 0]
                [wb, we] = [0, 0]
                for peid in range(MoraxConfig.PEArrayNum):  # 0 - num-1
                    pe_tasksize_tup = [0, 0]
                    if assigned_pearray + peid >= P:  # last outofrange pes
                        pe_tasksize_tup[0] = 0
                        pe_tasksize_tup[1] = 0
                    else:
                        cc = (assigned_pearray + peid) // (H * W)
                        hh = (assigned_pearray + peid) % (H * W) // W
                        ww = (assigned_pearray + peid) % (H * W) % W
                        if peid == 0:
                            [cb, ce] = [cc, cc]
                            [hb, he] = [hh, hh]
                            [wb, we] = [ww, ww]
                        else:
                            ce = cc if cc > cb else ce
                            he = hh
                            we = ww
                        if hh == H - 1 and H_tail > 0:
                            pe_tasksize_tup[0] = H_tail
                        else:
                            pe_tasksize_tup[0] = MoraxConfig.PEArraySize
                        if ww == W - 1 and W_tail > 0:
                            pe_tasksize_tup[1] = W_tail
                        else:
                            pe_tasksize_tup[1] = MoraxConfig.PEArraySize
                    tasksize_listoftup.append(pe_tasksize_tup)
                hbegin = hb * MoraxConfig.PEArraySize
                hend = (
                    omapsize - 1
                    if he == H - 1
                    else (he + 1) * MoraxConfig.PEArraySize - 1
                )
                wbegin = wb * MoraxConfig.PEArraySize
                wend = (
                    omapsize - 1
                    if we == W - 1
                    else (we + 1) * MoraxConfig.PEArraySize - 1
                )
                bulkscratch = {}
                bulkscratch["C"] = (cb + grp * group_outc, ce + grp * group_outc)
                bulkscratch["H"] = (hbegin, hend)
                bulkscratch["W"] = (wbegin, wend)
                listof_tasksize_listoftup.append(copy.deepcopy(tasksize_listoftup))
                listof_scratchdict.append(copy.deepcopy(bulkscratch))
                assigned_pearray += MoraxConfig.PEArrayNum
            # End spcify
            # Make Bulk
            for taskid in range(TSK):
                tasksize_listoftup = listof_tasksize_listoftup[taskid]
                ofbulkscratch = listof_scratchdict[taskid]
                chnum = len(ofbulkscratch["C"])
                # Make Weight Bulk
                wbulkscratch = {}
                wbulkscratch["K"] = ofbulkscratch["C"]
                wbulkscratch["RS"] = 0
                wbulkscratch["C"] = (grp * group_inc, (grp + 1) * group_inc - 1)
                wbulksize = (
                    _layerclass.kernel_size ** 2
                    * group_inc
                    * chnum
                    * MoraxConfig.PrecisionBits
                    // 8
                )
                wbulk = DataBulk(_modelname, _index, "WET", wbulksize, wbulkscratch,)
                qrw = QueryBuffer(wbulk, BO.Read, CC.WeightBuffer, CC.TensorCore)
                SubQueryList.append(copy.deepcopy(qrw))
                # Make Feature Bulk
                finsize = 0
                fosize = 0
                for pe_tasksize_tup in tasksize_listoftup:
                    finsize += (
                        pe_tasksize_tup[0] * _layerclass.stride
                        + _layerclass.kernel_size
                        - 1
                    ) * (
                        pe_tasksize_tup[1] * _layerclass.stride
                        + _layerclass.kernel_size
                        - 1
                    )
                    fosize += pe_tasksize_tup[0] * pe_tasksize_tup[1]
                ifbulkscratch = {}
                ifbulkscratch["C"] = (grp * group_inc, (grp + 1) * group_inc - 1)
                hbegin = (
                    0
                    if ofbulkscratch["H"][0] == 0
                    else ofbulkscratch["H"][0] * _layerclass.stride - 1
                )
                hend = (
                    _layerclass.feature_size - 1
                    if ofbulkscratch["H"][1] == omapsize - 1
                    else (
                        ofbulkscratch["H"][1] * _layerclass.stride
                        - 1
                        + _layerclass.kernel_size
                        - 1
                    )
                )
                wbegin = (
                    0
                    if ofbulkscratch["W"][0] == 0
                    else ofbulkscratch["W"][0] * _layerclass.stride - 1
                )
                wend = (
                    _layerclass.feature_size - 1
                    if ofbulkscratch["W"][1] == omapsize - 1
                    else (
                        ofbulkscratch["W"][1] * _layerclass.stride
                        - 1
                        + _layerclass.kernel_size
                        - 1
                    )
                )
                ifbulkscratch["H"] = (hbegin, hend)
                ifbulkscratch["W"] = (wbegin, wend)
                # todo check it
                for bat in range(B):
                    ifbulkscratch["B"] = bat
                    ofbulkscratch["B"] = bat
                    ifbulk = DataBulk(
                        _modelname,
                        _index + IIleft,
                        "FTR",
                        finsize * group_inc * MoraxConfig.PrecisionBits // 8,
                        ifbulkscratch,
                    )
                    qrf = QueryBuffer(ifbulk, BO.Read, ICtypeLeft, CC.TensorCore)
                    SubQueryList.append(copy.deepcopy(qrf))
                    # Make query
                    cmos_taskindex += 1
                    tctasklabel = make_tasklabel(
                        _modelname, _index, cmos_taskindex, "NGCONV"
                    )
                    tcbulksize = finsize * group_inc * MoraxConfig.PrecisionBits // 8
                    if bat == 0:
                        tcbulksize += wbulksize
                    qe = QueryExcuteOnTC(
                        _layerclass,
                        tctasklabel,
                        "OS",
                        layertype,
                        tasksize_listoftup,
                        tcbulksize,
                    )
                    SubQueryList.append(copy.deepcopy(qe))
                    # Make writeback query
                    wbbulk = DataBulk(
                        _modelname,
                        _index,
                        "FTR",
                        fosize * MoraxConfig.PrecisionBits // 8,
                        ofbulkscratch,
                        token,
                    )
                    qw = QueryBuffer(wbbulk, BO.Write, CC.TensorCore, CC.FeatureBuffer)
                    SubQueryList.append(copy.deepcopy(qw))
                SubQueryList.append(VritualQuerySeparator())
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
        TSK = math.ceil(chsize * 1.0 * H * W / MoraxConfig.PEArrayNum)
        P = math.ceil(chsize * 1.0 * H * W)  # ONETC
        H_tail = hsize % MoraxConfig.PEArraySize
        W_tail = wsize % MoraxConfig.PEArraySize
        B = _batch
        # Spcify os task
        assigned_pearray = 0
        listof_tasksize_listoftup = []
        listof_scratchdict = []
        while assigned_pearray < P:  # 1 - PE
            tasksize_listoftup = []
            [cb, ce] = [0, 0]
            [hb, he] = [0, 0]
            [wb, we] = [0, 0]
            for peid in range(MoraxConfig.PEArrayNum):  # 0 - num-1
                pe_tasksize_tup = [0, 0]
                if assigned_pearray + peid >= P:  # last outofrange pes
                    pe_tasksize_tup[0] = 0
                    pe_tasksize_tup[1] = 0
                else:
                    cc = (assigned_pearray + peid) // (H * W)
                    hh = (assigned_pearray + peid) % (H * W) // W
                    ww = (assigned_pearray + peid) % (H * W) % W
                    if peid == 0:
                        [cb, ce] = [cc, cc]
                        [hb, he] = [hh, hh]
                        [wb, we] = [ww, ww]
                    else:
                        ce = cc if cc > cb else ce
                        he = hh
                        we = ww
                    if hh == H - 1 and H_tail > 0:
                        pe_tasksize_tup[0] = H_tail
                    else:
                        pe_tasksize_tup[0] = MoraxConfig.PEArraySize
                    if ww == W - 1 and W_tail > 0:
                        pe_tasksize_tup[1] = W_tail
                    else:
                        pe_tasksize_tup[1] = MoraxConfig.PEArraySize
                tasksize_listoftup.append(pe_tasksize_tup)
            hbegin = hb * MoraxConfig.PEArraySize
            hend = hsize - 1 if he == H - 1 else (he + 1) * MoraxConfig.PEArraySize - 1
            wbegin = wb * MoraxConfig.PEArraySize
            wend = wsize - 1 if we == W - 1 else (we + 1) * MoraxConfig.PEArraySize - 1
            bulkscratch = {}
            # bulkscratch["B"] = ALL
            bulkscratch["C"] = (cb, ce) if chsize > 1 else 0
            bulkscratch["H"] = (hbegin, hend)
            bulkscratch["W"] = (wbegin, wend)
            listof_tasksize_listoftup.append(copy.deepcopy(tasksize_listoftup))
            listof_scratchdict.append(copy.deepcopy(bulkscratch))
            assigned_pearray += MoraxConfig.PEArrayNum
        # End spcify
        # Make Bulk
        # (RREW)
        for taskid in range(TSK):
            tasksize_listoftup = listof_tasksize_listoftup[taskid]
            bulkscratch = listof_scratchdict[taskid]
            chnum = len(bulkscratch["C"])
            # Make input Bulk
            fsize = 0
            for pe_tasksize_tup in tasksize_listoftup:
                fsize += pe_tasksize_tup[0] * pe_tasksize_tup[1]
            for bat in range(B):
                iobulkscratch = {}
                iobulkscratch["B"] = bat
                if layertype == LLT.Residual:
                    datatype = "FTR"
                    iobulkscratch["C"] = bulkscratch["C"]
                    iobulkscratch["H"] = bulkscratch["H"]
                    iobulkscratch["W"] = bulkscratch["W"]
                elif layertype == LLT.MADD:
                    datatype = "MAT"
                    iobulkscratch["M"] = bulkscratch["H"]
                    iobulkscratch["N"] = bulkscratch["W"]
                else:
                    raise DataError
                bulkleft = DataBulk(
                    _modelname,
                    _index + IIleft,
                    datatype,
                    fsize * MoraxConfig.PrecisionBits // 8,
                    iobulkscratch,
                )
                bulkright = DataBulk(
                    _modelname,
                    _index + IIright,
                    datatype,
                    fsize * MoraxConfig.PrecisionBits // 8,
                    iobulkscratch,
                )
                qrl = QueryBuffer(bulkleft, BO.Read, ICtypeLeft, CC.TensorCore)
                qrr = QueryBuffer(bulkright, BO.Read, ICtypeRight, CC.TensorCore)
                SubQueryList.append(copy.deepcopy(qrl))
                SubQueryList.append(copy.deepcopy(qrr))
                # Make query
                cmos_taskindex += 1
                tctasklabel = make_tasklabel(
                    _modelname, _index, cmos_taskindex, "Residual"
                )
                tcbulksize = fsize * 2 * MoraxConfig.PrecisionBits // 8
                qe = QueryExcuteOnTC(
                    _layerclass,
                    tctasklabel,
                    "OS",
                    layertype,
                    tasksize_listoftup,
                    tcbulksize,
                )
                SubQueryList.append(copy.deepcopy(qe))
                # Make Writeback
                wbbulk = DataBulk(
                    _modelname,
                    _index,
                    datatype,
                    fsize * MoraxConfig.PrecisionBits // 8,
                    iobulkscratch,
                    token,
                )
                qw = QueryBuffer(wbbulk, BO.Write, CC.TensorCore, CC.FeatureBuffer)
                SubQueryList.append(copy.deepcopy(qw))
            SubQueryList.append(VritualQuerySeparator())
        # End for

    if layertype == LLT.Batchnorm:
        # for: C B
        H = math.ceil(_layerclass.feature_size * 1.0 / MoraxConfig.PEArraySize)
        W = math.ceil(_layerclass.feature_size * 1.0 / MoraxConfig.PEArraySize)
        TSK = math.ceil(_layerclass.channel * 1.0 * H * W / MoraxConfig.PEArrayNum)
        P = _layerclass.channel * H * W
        H_tail = _layerclass.feature_size % MoraxConfig.PEArraySize
        W_tail = H_tail
        B = _batch
        # Spcify os task
        assigned_pearray = 0
        listof_tasksize_listoftup = []
        listof_scratchdict = []
        while assigned_pearray < P:  # 1 - PE
            tasksize_listoftup = []
            [cb, ce] = [0, 0]
            [hb, he] = [0, 0]
            [wb, we] = [0, 0]
            for peid in range(MoraxConfig.PEArrayNum):  # 0 - num-1
                pe_tasksize_tup = [0, 0]
                if assigned_pearray + peid >= P:  # last outofrange pes
                    pe_tasksize_tup[0] = 0
                    pe_tasksize_tup[1] = 0
                else:
                    cc = (assigned_pearray + peid) // (H * W)
                    hh = (assigned_pearray + peid) % (H * W) // W
                    ww = (assigned_pearray + peid) % (H * W) % W
                    if peid == 0:
                        [cb, ce] = [cc, cc]
                        [hb, he] = [hh, hh]
                        [wb, we] = [ww, ww]
                    else:
                        ce = cc if cc > cb else ce
                        he = hh
                        we = ww
                    if hh == H - 1 and H_tail > 0:
                        pe_tasksize_tup[0] = H_tail
                    else:
                        pe_tasksize_tup[0] = MoraxConfig.PEArraySize
                    if ww == W - 1 and W_tail > 0:
                        pe_tasksize_tup[1] = W_tail
                    else:
                        pe_tasksize_tup[1] = MoraxConfig.PEArraySize
                tasksize_listoftup.append(pe_tasksize_tup)
            hbegin = hb * MoraxConfig.PEArraySize
            hend = (
                _layerclass.feature_size - 1
                if he == H - 1
                else (he + 1) * MoraxConfig.PEArraySize - 1
            )
            wbegin = wb * MoraxConfig.PEArraySize
            wend = (
                _layerclass.feature_size - 1
                if we == W - 1
                else (we + 1) * MoraxConfig.PEArraySize - 1
            )
            bulkscratch = {}
            bulkscratch["C"] = (cb, ce)
            bulkscratch["H"] = (hbegin, hend)
            bulkscratch["W"] = (wbegin, wend)
            listof_tasksize_listoftup.append(copy.deepcopy(tasksize_listoftup))
            listof_scratchdict.append(copy.deepcopy(bulkscratch))
            assigned_pearray += MoraxConfig.PEArrayNum
        # End spcify
        # [(L)(REW)]
        lut_taskindex = -1
        for taskid in range(TSK):
            tasksize_listoftup = listof_tasksize_listoftup[taskid]
            bulkscratch = listof_scratchdict[taskid]
            chnum = len(bulkscratch["C"])
            # Make LUT Query
            for chn in range(bulkscratch["C"][0], bulkscratch["C"][1] + 1, 1):
                lut_taskindex += 1
                luttasklabel = make_tasklabel(_modelname, _index, lut_taskindex, "LUT")
                lutadress = get_lookup_adress(
                    layertype, chn
                )  # tup3(clstid, nvtcid, sliceid)
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
            fsize = 0
            for pe_tasksize_tup in tasksize_listoftup:
                fsize += pe_tasksize_tup[0] * pe_tasksize_tup[1]
            for bat in range(B):
                # Make input Bulk
                bulkscratch["B"] = bat
                inbulk = DataBulk(
                    _modelname,
                    _index + IIleft,
                    "FTR",
                    fsize * MoraxConfig.PrecisionBits // 8,
                    bulkscratch,
                )
                qr = QueryBuffer(inbulk, BO.Read, ICtypeLeft, CC.TensorCore)
                SubQueryList.append(copy.deepcopy(qr))
                # Make query
                cmos_taskindex += 1
                tctasklabel = make_tasklabel(
                    _modelname, _index, cmos_taskindex, "Batchnorm"
                )
                tcbulksize = fsize * MoraxConfig.PrecisionBits // 8
                qe = QueryExcuteOnTC(
                    _layerclass,
                    tctasklabel,
                    "OS",
                    layertype,
                    tasksize_listoftup,
                    tcbulksize,
                )
                SubQueryList.append(copy.deepcopy(qe))
                # Make Writeback
                wbbulk = DataBulk(
                    _modelname, _index, "FTR", tcbulksize, bulkscratch, token,
                )
                qw = QueryBuffer(wbbulk, BO.Write, CC.TensorCore, CC.FeatureBuffer)
                SubQueryList.append(copy.deepcopy(qw))
            SubQueryList.append(VritualQuerySeparator())
        # End for

    if layertype == LLT.Layernorm:
        # for: B MN
        H = math.ceil(_layerclass.row_dim * 1.0 / MoraxConfig.PEArraySize)
        W = math.ceil(_layerclass.col_dim * 1.0 / MoraxConfig.PEArraySize)
        H_tail = _layerclass.row_dim % MoraxConfig.PEArraySize
        W_tail = _layerclass.col_dim % MoraxConfig.PEArraySize
        TSK = math.ceil(H * W / MoraxConfig.PEArrayNum)
        P = H * W
        B = _batch
        # one batch task
        assigned_pearray = 0
        listof_tasksize_listoftup = []
        listof_scratchdict = []
        while assigned_pearray < P:  # 1 - PE
            tasksize_listoftup = []
            [hb, he] = [0, 0]
            [wb, we] = [0, 0]
            for peid in range(MoraxConfig.PEArrayNum):  # 0 - num-1
                pe_tasksize_tup = [0, 0]
                if assigned_pearray + peid >= P:  # last outofrange pes
                    pe_tasksize_tup[0] = 0
                    pe_tasksize_tup[1] = 0
                else:
                    hh = (assigned_pearray + peid) // W
                    ww = (assigned_pearray + peid) % W
                    if peid == 0:
                        [hb, he] = [hh, hh]
                        [wb, we] = [ww, ww]
                    else:
                        he = hh
                        we = ww
                    if hh == H - 1 and H_tail > 0:
                        pe_tasksize_tup[0] = H_tail
                    else:
                        pe_tasksize_tup[0] = MoraxConfig.PEArraySize
                    if ww == W - 1 and W_tail > 0:
                        pe_tasksize_tup[1] = W_tail
                    else:
                        pe_tasksize_tup[1] = MoraxConfig.PEArraySize
                tasksize_listoftup.append(pe_tasksize_tup)
            hbegin = hb * MoraxConfig.PEArraySize
            hend = (
                _layerclass.row_dim - 1
                if he == H - 1
                else (he + 1) * MoraxConfig.PEArraySize - 1
            )
            wbegin = wb * MoraxConfig.PEArraySize
            wend = (
                _layerclass.col_dim - 1
                if we == W - 1
                else (we + 1) * MoraxConfig.PEArraySize - 1
            )
            bulkscratch = {}
            bulkscratch["M"] = (hbegin, hend)
            bulkscratch["N"] = (wbegin, wend)
            listof_tasksize_listoftup.append(copy.deepcopy(tasksize_listoftup))
            listof_scratchdict.append(copy.deepcopy(bulkscratch))
            assigned_pearray += MoraxConfig.PEArrayNum
        # End spcify
        # Make bulk
        # [L(REW)]
        lut_taskindex = -1
        for bat in range(B):
            # Make LUT
            lut_taskindex += 1
            luttasklabel = make_tasklabel(_modelname, _index, lut_taskindex, "LUT")
            lutadress = get_lookup_adress(
                layertype, bat
            )  # tup3(clstid, nvtcid, sliceid)
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
            for taskid in range(TSK):  # ok, tsk size various within one bulk
                tasksize_listoftup = listof_tasksize_listoftup[taskid]
                bulkscratch = listof_scratchdict[taskid]
                bulkscratch["B"] = bat
                # Make Input Bulk
                fsize = 0
                for pe_tasksize_tup in tasksize_listoftup:
                    fsize += pe_tasksize_tup[0] * pe_tasksize_tup[1]
                inbulk = DataBulk(
                    _modelname,
                    _index + IIleft,
                    "MAT",
                    fsize * MoraxConfig.PrecisionBits // 8,
                    bulkscratch,
                )
                qr = QueryBuffer(inbulk, BO.Read, ICtypeLeft, CC.TensorCore)
                SubQueryList.append(copy.deepcopy(qr))
                # Make query
                cmos_taskindex += 1
                tctasklabel = make_tasklabel(
                    _modelname, _index, cmos_taskindex, "Layernorm"
                )
                tcbulksize = fsize * MoraxConfig.PrecisionBits // 8
                qe = QueryExcuteOnTC(
                    _layerclass,
                    tctasklabel,
                    "OS",
                    layertype,
                    tasksize_listoftup,
                    tcbulksize,
                )
                SubQueryList.append(copy.deepcopy(qe))
                # Make Writeback
                wbbulk = DataBulk(
                    _modelname, _index, "MAT", tcbulksize, bulkscratch, token,
                )
                qw = QueryBuffer(wbbulk, BO.Write, CC.TensorCore, CC.FeatureBuffer)
                SubQueryList.append(copy.deepcopy(qw))
            SubQueryList.append(VritualQuerySeparator())
            # End for

    if layertype == LLT.GEMM:
        if IIleft == 0:
            n_dim = _layerclass.m_dim
            m_dim = _layerclass.k_dim
            k_dim = _layerclass.n_dim
        else:
            k_dim = _layerclass.k_dim
            n_dim = _layerclass.n_dim
            m_dim = _layerclass.m_dim
        H = math.ceil(m_dim * 1.0 / MoraxConfig.PEArraySize)
        W = math.ceil(n_dim * 1.0 / MoraxConfig.PEArraySize)
        TSK = math.ceil(H * W * 1.0 / MoraxConfig.PEArrayNum)
        P = H * W
        H_tail = m_dim % MoraxConfig.RRAMXbarSize
        W_tail = n_dim % MoraxConfig.RRAMXbarSize
        B = _batch
        # BMN
        # one batch task
        assigned_pearray = 0
        listof_tasksize_listoftup = []
        listof_scratchdict = []
        while assigned_pearray < P:  # 1 - PE
            tasksize_listoftup = []
            [hb, he] = [0, 0]
            [wb, we] = [0, 0]
            for peid in range(MoraxConfig.PEArrayNum):  # 0 - num-1
                pe_tasksize_tup = [0, 0]
                if assigned_pearray + peid >= P:  # last outofrange pes
                    pe_tasksize_tup[0] = 0
                    pe_tasksize_tup[1] = 0
                else:
                    hh = (assigned_pearray + peid) // W
                    ww = (assigned_pearray + peid) % W
                    if peid == 0:
                        [hb, he] = [hh, hh]
                        [wb, we] = [ww, ww]
                    else:
                        he = hh
                        we = ww
                    if hh == H - 1 and H_tail > 0:
                        pe_tasksize_tup[0] = H_tail
                    else:
                        pe_tasksize_tup[0] = MoraxConfig.PEArraySize
                    if ww == W - 1 and W_tail > 0:
                        pe_tasksize_tup[1] = W_tail
                    else:
                        pe_tasksize_tup[1] = MoraxConfig.PEArraySize
                tasksize_listoftup.append(pe_tasksize_tup)
            hbegin = hb * MoraxConfig.PEArraySize
            hend = m_dim - 1 if he == H - 1 else (he + 1) * MoraxConfig.PEArraySize - 1
            wbegin = wb * MoraxConfig.PEArraySize
            wend = n_dim - 1 if we == W - 1 else (we + 1) * MoraxConfig.PEArraySize - 1
            bulkscratch = {}
            bulkscratch["M"] = (hbegin, hend)
            bulkscratch["N"] = (wbegin, wend)
            listof_tasksize_listoftup.append(copy.deepcopy(tasksize_listoftup))
            listof_scratchdict.append(copy.deepcopy(bulkscratch))
            assigned_pearray += MoraxConfig.PEArrayNum
        # End spcify
        # Make GEMM innerplace Bulk
        # MNB, [R/ (R/REW)]
        for taskid in range(TSK):
            tasksize_listoftup = listof_tasksize_listoftup[taskid]
            obulkscratch = listof_scratchdict[taskid]
            leftbulkscratch = {}
            rightbulkscratch = {}
            leftbulkscratch["M"] = (
                (obulkscratch["M"][0], obulkscratch["M"][1])
                if (obulkscratch["M"][0] < obulkscratch["M"][1])
                else (0, m_dim - 1)
            )
            leftbulkscratch["N"] = (0, k_dim - 1)
            rightbulkscratch["M"] = (0, k_dim - 1)
            leftbulkscratch["N"] = (
                (obulkscratch["N"][0], obulkscratch["N"][1])
                if (obulkscratch["N"][0] < obulkscratch["N"][1])
                else (0, n_dim - 1)
            )
            insize_l = (
                k_dim
                * (leftbulkscratch["M"][1] - leftbulkscratch["M"][0] + 1)
                * MoraxConfig.PrecisionBits
                / 8
            )
            insize_r = (
                k_dim
                * (rightbulkscratch["N"][1] - rightbulkscratch["N"][0] + 1)
                * MoraxConfig.PrecisionBits
                / 8
            )
            # for pe_tasksize_tup in tasksize_listoftup:
            #     finsize_l += pe_tasksize_tup[0] * k_dim
            #     finsize_r += pe_tasksize_tup[1] * k_dim
            # Make Weight Bulk if had
            if IIleft == 0:
                leftbulkscratch["B"] = 0
                leftbulk = DataBulk(
                    _modelname, _index + IIleft, "MAT", insize_l, leftbulkscratch,
                )
                qrl = QueryBuffer(leftbulk, BO.Read, ICtypeLeft, CC.TensorCore)
                SubQueryList.append(copy.deepcopy(qrl))
            elif IIright == 0:
                rightbulkscratch["B"] = 0
                rightbulk = DataBulk(
                    _modelname, _index + IIright, "MAT", insize_r, rightbulkscratch,
                )
                qrr = QueryBuffer(rightbulk, BO.Read, ICtypeRight, CC.TensorCore)
                SubQueryList.append(copy.deepcopy(qrr))
            # Make input bulk
            for bat in range(B):
                if IIleft < 0:
                    leftbulkscratch["B"] = bat
                    leftbulk = DataBulk(
                        _modelname, _index + IIleft, "MAT", insize_l, leftbulkscratch,
                    )
                    qrl = QueryBuffer(leftbulk, BO.Read, ICtypeLeft, CC.TensorCore)
                    SubQueryList.append(copy.deepcopy(qrl))
                elif IIright < 0:
                    rightbulkscratch["B"] = bat
                    rightbulk = DataBulk(
                        _modelname, _index + IIright, "MAT", insize_r, rightbulkscratch,
                    )
                    qrr = QueryBuffer(rightbulk, BO.Read, ICtypeRight, CC.TensorCore)
                    SubQueryList.append(copy.deepcopy(qrr))
                # Make query
                cmos_taskindex += 1
                tctasklabel = make_tasklabel(_modelname, _index, cmos_taskindex, "GEMM")
                # Get bulksize
                if IIleft == 0 and IIright < 0:
                    tcbulksize = insize_r
                    if bat == 0:
                        tcbulksize += insize_l
                elif IIleft < 0 and IIright == 0:
                    tcbulksize = insize_l
                    if bat == 0:
                        tcbulksize += insize_r
                else:
                    tcbulksize = insize_r + insize_l
                qe = QueryExcuteOnTC(
                    _layerclass,
                    tctasklabel,
                    "OS",
                    layertype,
                    tasksize_listoftup,
                    tcbulksize,
                )
                SubQueryList.append(copy.deepcopy(qe))
                # Make writeback query
                fosize = 0
                for tup in tasksize_listoftup:
                    fosize += tup[0] * tup[1]
                wbbulk = DataBulk(
                    _modelname,
                    _index,
                    "MAT",
                    fosize * MoraxConfig.PrecisionBits // 8,
                    obulkscratch,
                    token,
                )
                qw = QueryBuffer(wbbulk, BO.Write, CC.TensorCore, CC.FeatureBuffer)
                SubQueryList.append(copy.deepcopy(qw))
            SubQueryList.append(VritualQuerySeparator())
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
        ofsize = _layerclass.feature_size // _layerclass.kernel_size
        # [REW]
        for bat in range(B):
            for ch in range(C):
                # rf
                datatype = "FTR"
                fbulkscratch = {}
                fbulkscratch["B"] = bat
                fbulkscratch["H"] = (0, _layerclass.feature_size - 1)
                fbulkscratch["W"] = (0, _layerclass.feature_size - 1)
                fbulkscratch["C"] = ch
                fbs = _layerclass.feature_size ** 2 * MoraxConfig.PrecisionBits // 8
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
                    _modelname, _index, vpu_taskindex, "Pooling"
                )
                qv = QueryExcuteOnVPU(_layerclass, vtasklabel, "Linear", NLT.Pooling,)
                SubQueryList.append(copy.deepcopy(qv))
                # wb
                wbs = ofsize ** 2 * MoraxConfig.PrecisionBits // 8
                wbulkscratch = {
                    "B": bat,
                    "C": ch,
                    "H": (0, ofsize - 1),
                    "W": (0, ofsize - 1),
                }
                wbbulk = DataBulk(_modelname, _index, datatype, wbs, wbulkscratch,)
                qwb = QueryBuffer(wbbulk, BO.Write, CC.VPU, CC.FeatureBuffer)
                SubQueryList.append(copy.deepcopy(qwb))
            SubQueryList.append(VritualQuerySeparator())
        # End for

    if layertype == NLT.Softmax1D or NLT.Softmax2D:
        B = _batch
        L = 1 if layertype == NLT.Softmax1D else _layerclass.row_dim
        V = _layerclass.v_dim if layertype == NLT.Softmax1D else _layerclass.col_dim
        # [REE(L)LEW]
        for bat in range(B):
            for line in range(L):
                # read
                rbulkscratch = {}
                rbulkscratch["B"] = bat
                if layertype == NLT.Softmax2D:
                    datatype = "MAT"
                    rbulkscratch["M"] = line
                    rbulkscratch["N"] = (0, V)
                elif layertype == NLT.Softmax1D:
                    datatype = "VEC"
                    rbulkscratch["M"] = (0, V)
                rbsize = V * MoraxConfig.PrecisionBits // 8
                rbulk = DataBulk(
                    _modelname,
                    _index + _layerclass.input_indecies_tuple[0],
                    datatype,
                    rbsize,
                    rbulkscratch,
                )
                qr = QueryBuffer(rbulk, BO.Read, CC.FeatureBuffer, CC.VPU)
                SubQueryList.append(copy.deepcopy(qr))
                # vmax
                vpu_taskindex += 1
                vtasklabel = make_tasklabel(
                    _modelname, _index, vpu_taskindex, "Softmax"
                )
                qv = QueryExcuteOnVPU(_layerclass, vtasklabel, "Softmax", SO.VMAX,)
                SubQueryList.append(copy.deepcopy(qv))
                # SO.Truncation
                smu_taskindex += 1
                stasklabel = make_tasklabel(
                    _modelname, _index, smu_taskindex, "Softmax"
                )
                qsmu = QueryExcuteOnSMU(
                    _layerclass, stasklabel, "UpStream", SO.Truncation,
                )
                SubQueryList.append(copy.deepcopy(qsmu))
                # exp lookup
                for v in range(V):
                    lut_taskindex += 1
                    luttasklabel = make_tasklabel(
                        _modelname, _index, lut_taskindex, "LUT"
                    )
                    lutadress = get_lookup_adress(_index, line)
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
                luttasklabel = make_tasklabel(_modelname, _index, lut_taskindex, "LUT")
                lutadress = get_lookup_adress(_index, line)
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
                    _modelname, _index, vpu_taskindex, "Softmax"
                )
                qv = QueryExcuteOnVPU(_layerclass, vtasklabel, "Softmax", SO.VNORM,)
                SubQueryList.append(copy.deepcopy(qv))
                # write
                wbulk = DataBulk(_modelname, _index, datatype, rbsize, rbulkscratch,)
                qw = QueryBuffer(wbulk, BO.Write, CC.VPU, CC.FeatureBuffer)
                SubQueryList.append(copy.deepcopy(qw))
        # Eno for
    return SubQueryList


def make_tc_task_list(_tasklen):
    ttltup = []
    for peid in range(MoraxConfig.PEArrayNum):
        if _tasklen >= MoraxConfig.PEArraySize:
            _tasklen -= MoraxConfig.PEArraySize
            ttltup.append((MoraxConfig.PEArraySize, MoraxConfig.PEArraySize))
        elif _tasklen > 0:
            _tasklen -= MoraxConfig.PEArraySize
            ttltup.append((MoraxConfig.PEArraySize, _tasklen))
        else:
            ttltup.append((0, 0))
    return ttltup
