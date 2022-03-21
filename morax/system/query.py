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
from hardware.cluster import HWDicts

# [bulk]
# indicate the data form of input and weight
# bulkfrom = (part or whole) feature: NCHW  weight: KCRS  MVM & GEMM: TODO
# dataname = W or F or TODO
# bulklable = modelname_'L'+layeridx_'dataname'+bulkidx_bulksizekB_bulkfrom

# [task]
# indicate the task excution scale using [output]
# taskform = (part or whole) batchN outputchannelC heighH widthW  MVM: outputdimO batchN GEMM: heightH widthW batchN
# CONV
# RRAM: CHWN onefeaturebyonefeature    OS: HWNC onechannelbyonechannel (FOR MAX KERNEL REUSE)
# TODO
# tasklable = modelname_'L'+layeridx_'T'+taskidx_taskform


class SubQueryRead:
    def __init__(self, _bulksize, _bulklabel, _locationEnum, _toEnum):
        self.bulksize = _bulksize
        self.bulklabel = _bulklabel
        self.locationEnum = _locationEnum
        self.toEnum = _toEnum


class SubQueryWrite:
    def __init__(self, _bulksize, _bulklabel, _locationEnum, _toEnum):
        self.bulksize = _bulksize
        self.bulklabel = _bulklabel
        self.locationEnum = _locationEnum
        self.toEnum = _toEnum


class SubQueryExcute:
    def __init__(self, _layertype, _tasklabel: str):
        self.tasklabel = _tasklabel
        self.layertype = _layertype
        self.taskfrom = self.get_taskform(_tasklabel)

    def get_taskform(self, _tasklabel):
        form = _tasklabel[_tasklabel.rfind("_") + 1 :]
        return form


class SubQueryExcuteOnTC(SubQueryExcute):
    def __init__(
        self, _layertype, _tasklabel: str, _dfmod: str, _execution, _tasksize: float
    ):
        super().__init__(_layertype, _tasklabel)
        self.dfmod = _dfmod
        self.execution = _execution
        self.tasksize = _tasksize
        assert self.checkquery() is True

    def checkquery(self):
        dfsafe = self.dfmod == "OS" or self.dfmod == "Systolic"
        executionsafe = (
            self.execution in MoraxExecutionDict[ClusterComponent.TensorCore]
        )
        tasksafe = self.tasksize <= HWDicts["PEArrayNum"]
        return dfsafe and executionsafe and tasksafe


class SubQueryExcuteOnNVTC(SubQueryExcute):
    def __init__(
        self, _layertype, _tasklabel: str, _dfmod: str, _execution, _tasksize: float
    ):
        super().__init__(_layertype, _tasklabel)
        self.dfmod = _dfmod
        self.execution = _execution
        self.tasksize = _tasksize
        assert self.checkquery() is True

    def checkquery(self):
        dfsafe = self.dfmod == "Xbar" or self.dfmod == "LookUp"
        executionsafe = (
            self.execution in MoraxExecutionDict[ClusterComponent.nvTensorCore]
        )
        tasksafe = self.tasksize <= HWDicts["RRAMSliceNum"]
        return dfsafe and executionsafe and tasksafe


class SubQueryExcuteOnVPU(SubQueryExcute):
    def __init__(
        self, _layertype, _tasklabel: str, _dfmod: str, _execution, _tasksize: float
    ):
        super().__init__(_layertype, _tasklabel)
        self.dfmod = _dfmod
        self.execution = _execution
        self.tasksize = _tasksize
        assert self.checkquery() is True

    def checkquery(self):
        dfsafe = self.dfmod == "Para" or self.dfmod == "Reduce"
        executionsafe = self.execution in MoraxExecutionDict[ClusterComponent.SMU]
        tasksafe = self.tasksize <= HWDicts["LaneNum"]
        return dfsafe and executionsafe and tasksafe


class SubQueryExcuteOnSMU(SubQueryExcute):
    def __init__(
        self, _layertype, _tasklabel: str, _dfmod: str, _execution, _tasksize: float
    ):
        super().__init__(_layertype, _tasklabel)
        self.dfmod = _dfmod
        self.execution = _execution
        self.tasksize = _tasksize
        assert self.checkquery() is True

    def checkquery(self):
        dfsafe = self.dfmod == "UpStream" or self.dfmod == "DownStream"
        executionsafe = self.execution in MoraxExecutionDict[ClusterComponent.SMU]
        tasksafe = self.tasksize <= HWDicts["SMUMaxIO"]
        return dfsafe and executionsafe and tasksafe


class SubQueryClusterTransfer:
    def __init__(self, _bulklabel, _bulksize, _fromCluster, _toCluster):
        self.bulksize = _bulksize
        self.bulklabel = _bulklabel
        self.fromCluster = _fromCluster
        self.toCluster = _toCluster


class SubQueryClusterTransfer:
    def __init__(self, _bulklabel, _bulksize, _toCluster):
        self.bulksize = _bulksize
        self.bulklabel = _bulklabel
        self.toCluster = _toCluster


# _edges: [vertex1, (vertex2, ...)]
# vertex: {'idx': , 'type': , 'location': }
class Query:
    def __init__(self, _layerclass, _location, _edges) -> None:
        self.index = _layerclass.layer_index
        self.query


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
