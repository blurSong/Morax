# system query class
# Morax means the mixing of CMOS and RRAM
# Author: Tao Song.
# Last modification: 0316


from sqlite3 import DataError
import queue
import sys
import numpy as np
import copy
import pandas as pd
import subprocess as SP
from morax.system.interface import *
from morax.model.layer import LinearLayerType as LLT, NonlinearLayerType as NLT, mxLTD, mxNTD
from morax.system.config import MoraxConfig, HWParam
from morax.hardware.buffer import DataBulk
from morax.hardware.chip import MoraxChip
import mapper
import math
from morax.model.model import ModelDAG, ModelList, ModelType
from morax.frontend.api import get_idx_from_concat

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
        _tasksize: tuple(int, int),
    ):
        super().__init__(_layerclass, _tasklabel)
        self.dfmod = _dfmod
        self.execution = _execution
        self.tasksize = _tasksize  #  (rowparts, collines)
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
        _tasksize: tuple(int, int),
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


""" ================================================================
## generate layer query and compilie them to sub query
====================================================================
"""


class LayerQuery:
    def __init__(
        self, _q_index: int, _batch: int, _layerclass, _assignment, _indegree, _outdegree
    ) -> None:
        self.q_index = _q_index
        self.batch = _batch
        self.assignment = _assignment
        self.layerclass = copy.deepcopy(_layerclass)
        self.iodegree = {'in': _indegree, 'out': _outdegree}
        self.SubQueryList = []

        # assignment: listoftuple_clstid_nvtcid_sliceidlist
        # default []
        if isinstance(self.assignment, list):
            assert mapper.check_mapping_with_query(
                self.layerclass.layer_index
            ), "{}".format(self.layerclass.layer_name)

    def compile(self, _modelname, moraxchip: MoraxChip):
        # Generate subqueries of this layer query
        print("[Morax][System] Compiling Query {}.".format(self.q_index))
        layertype = self.layerclass.layer_type
        tc = MoraxConfig.TCNum
        pearray = MoraxConfig.PEArrayNum
        pesize = MoraxConfig.PEArraySize
        if layertype == "Linear":
            if self.assignment:  # on NVTC
                self.SubQueryList.append(copy.deepcopy(compileRRAM(self.q_index, _modelname, self.layerclass, self.assignment, moraxchip, self.batch, self.iodegree['out'])))
            else:
                self.SubQueryList.append(copy.deepcopy(compileCMOS(self.q_index, _modelname, self.layerclass, moraxchip, self.batch)))

        # Read
        # Excute
        # WriteBack
        return



def make_tasklabel(mn, li, ti, bf: str):
    return mn + "_L" + str(li) + "T" + str(ti) + "_" + bf


def compileRRAM(_index, _modelname,  _layerclass, _doclotnsl: dict, _chip: MoraxChip, _batch, token):
    # dict of clstid: [tuple1(nvtcid, sliceidlist), tuple2, ... ]
    SubQueryList = []
    layertype = _layerclass.layer_type

    if mxLTD[layertype] == "Linear":
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
                        mapinfo.append(_chip.ClusterList[clstid].nvTensorCoreList[nvtcid].RRAMSliceObjList[info].layerinfo[1])
                    bulkinfo = []
                    for tup in mapinfo:
                        if tup[0] not in bulkinfo:
                            bulkinfo.append(tup[0])
                    # make query
                    datatype = 'FTR'
                    bulkscratch = {}
                    taskindex += 1
                    bulkscratch['B'] = bat
                    bulkscratch['HW'] = 19940117
                    bulkscratch['C'] = []
                    bs = 0
                    for inf in bulkinfo:
                        if inf == M and M_tail > 0:
                            bs += M_tail
                        else:
                            bs += MoraxConfig.RRAMXbarSize
                    bulkscratch['C'].append(inf)
                    bs = bs * MoraxConfig.PrecisionBits / 8
                    bulk = DataBulk(_modelname, _index, datatype, bs, bulkscratch)
                    qr = QueryBuffer(bulk, BO.Read, CC.FeatureBuffer, CC.nvTensorCore)
                    tasklabel = make_tasklabel(_modelname, _index, taskindex, "Linear")
                    qe = QueryExcuteOnNVTC(_layerclass, tasklabel, "Xbar", LLT.Linear, nvtcid, clstid, sliceidlist)
                    SubQueryList.append(copy.deepcopy(qr))
                    SubQueryList.append(copy.deepcopy(qe))
            tasklabel = make_tasklabel(_modelname, _index, 0, "PostProcess")
            qv = QueryExcuteOnVPU(_layerclass, tasklabel, 'PostProcess', LLT.Linear, (M, _layerclass.col_dim))
            SubQueryList.append(copy.deepcopy(qv))
            bulk = DataBulk(_modelname, _index, 'FTR', _layerclass.col_dim * MoraxConfig.PrecisionBits / 8, {'B': bat, 'HW': 19940117, 'C': 19940117}, token)
            qw = QueryBuffer(bulk, BO.Write, CC.VPU, CC.FeatureBuffer)
            SubQueryList.append(copy.deepcopy(qw))
        # End for

    if mxLTD[layertype] == "CONV":
        M = math.ceil(_layerclass.in_channel * _layerclass.kernel_size ** 2 * 1.0 / MoraxConfig.RRAMXbarSize)
        N = math.ceil(_layerclass.out_channel * 1.0 / MoraxConfig.RRAMXbarSize)
        M_tail = _layerclass.in_channel * _layerclass.kernel_size ** 2 % MoraxConfig.RRAMXbarSize
        N_tail = _layerclass.colout_channel_dim % MoraxConfig.RRAMXbarSize
        rram_taskindex = -1
        vpu_taskindex = -1
        for bat in range(_batch):
            omapsize = _layerclass.feature_size / _layerclass.stride
            for row_iter in range(omapsize):
                for col_iter in range(omapsize):
                    # make simple bulk
                    datatype = "FTR"
                    bulkscratch = {}
                    bulkscratch['B'] = bat
                    bulkscratch['HW'] = col_iter + row_iter * omapsize
                    bulkscratch['C'] = 19940117
                    bs = _layerclass.kernel_size * _layerclass.in_channel * MoraxConfig.PrecisionBits / 8
                    bs *= _layerclass.kernel_size if col_iter == 0 else _layerclass.stride
                    bulk = DataBulk(_modelname, _index, datatype, bs, bulkscratch)
                    qr = QueryBuffer(bulk, BO.Read, CC.FeatureBuffer, CC.nvTensorCore)
                    SubQueryList.append(copy.deepcopy(qr))
                    # make subexe query
                    # dict of clstid: [tuple1(nvtcid, sliceidlist), tuple2, ... ]
                    for clstid, nvtclist in _doclotnsl.items():
                        for nvtctup in nvtclist:
                            (nvtcid, sliceidlist) = nvtctup
                            rram_taskindex += 1
                            rtasklabel = make_tasklabel(_modelname, _index, rram_taskindex, "CONV")
                            qe = QueryExcuteOnNVTC(_layerclass, rtasklabel, "Xbar", LLT.CONV, nvtcid, clstid, sliceidlist)
                            SubQueryList.append(copy.deepcopy(qe))
                    vpu_taskindex += 1
                    vtasklabel = make_tasklabel(_modelname, _index, vpu_taskindex, "PostProcess")
                    qv = QueryExcuteOnVPU(_layerclass, vtasklabel, 'PostProcess', LLT.CONV, (M, _layerclass.out_channel))
                    SubQueryList.append(copy.deepcopy(qv))
                    # make writeback bulk
                    bulk = DataBulk(_modelname, _index, 'FTR', _layerclass.out_channel * MoraxConfig.PrecisionBits / 8, {'B': bat, 'HW': col_iter + row_iter * omapsize, 'C': 19940117}, token)
                    qw = QueryBuffer(bulk, BO.Write, CC.VPU, CC.FeatureBuffer)
                    SubQueryList.append(copy.deepcopy(qw))
            # End for








def compileCMOS(layerclass, moraxchip: MoraxChip, batch):




def generate_queries(
    _modelList: ModelList, _modelDAG: ModelDAG, _modeltype, concatlist=[], _batch=1
):
    # _modelDAG to one sequential queryList using BFS
    layernumL = _modelList.layernum
    layernumG = _modelDAG.layernum
    if layernumL != layernumG and _modeltype != ModelType.MHATTENTION:
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
                idx, _batch, _modelList[idx], _modelDAG.LayerAssignmentDict[idx], len(_modelDAG.fromVertexDict[idx]), len(_modelDAG.toVertexDict[idx])
            )
        else:
            oidx = get_idx_from_concat(idx, concatlist)
            q = LayerQuery(
                idx, _batch, _modelList[oidx], _modelDAG.LayerAssignmentDict[idx], len(_modelDAG.fromVertexDict[idx]), len(_modelDAG.toVertexDict[idx])
            )
        # q.compile()
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
