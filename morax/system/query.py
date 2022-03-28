# system query class
# Morax means the mixing of CMOS and RRAM
# Author: Tao Song.
# Last modification: 0316

import os
import re
import sys
import numpy as np
import copy
import pandas as pd
import subprocess as SP
from morax.system.interface import MoraxExecutionDict, ClusterComponent
from morax.model.layer import LinearLayerType as LLT, NonlinearLayerType as NLT
from morax.system.config import MoraxConfig, HWParam
from morax.hardware.buffer import DataBulk

# [bulk]
# indicate the data form of input and weight
# bulkfrom = (part or whole) feature: NCHW  weight: KCRS  MVM & GEMM: TODO
# dataname = W or F or TODO
# bulklabel = modelname_'L'+layeridx_'dataname'+bulkidx_bulksizeByte_bulkfrom

# [task]
# indicate the task excution scale using [output]
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
        _sliceidlist,
        # _tasksize: tuple(int, int),
    ):
        super().__init__(_layerclass, _tasklabel)
        self.dfmod = _dfmod
        self.execution = _execution
        self.nvtcid = _nvtcid
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


# edges: [vertex1, (vertex2, ...)]
# vertex: {'idx': , 'type': , 'location': }
class LayerQuery:
    def __init__(self, _layerclass, _location, _edges) -> None:
        self.index = _layerclass.layer_index
        self.querylist = []

    def compile_query():
        # generate subqueries of this layer
        return


def generate_queries(_model_list, _assignment_list):
    layernum = len(_model_list)
    assert layernum == len(_assignment_list)


def generate_queries_mt(
    _model_list1, _model_list2, _assignment_list1, _assignment_list2
):
    layernum1 = len(_model_list1)
    layernum2 = len(_model_list2)
    assert layernum1 == len(_assignment_list1)
    assert layernum2 == len(_assignment_list2)
