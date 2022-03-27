# VPU class of Morax
# Morax means the mixing of CMOS and RRAM
# Author: Tao Song.
# Last modification: 0326

import numpy as np
import subprocess as SP
import multiprocessing as MP
import copy
import math
from morax.system.interface import MoraxExecutionDict, ClusterComponent
from morax.system.timefilm import TimeFilm, TimeStamp
from collections import UserDict, UserList
from morax.system.config import MoraxConfig
from morax.system.query import QueryExcuteOnVPU
from morax.model.layer import Softmax1D, Softmax2D

VPUExe = MoraxExecutionDict[ClusterComponent.VPU]


class VPUActionDict:
    def __init__(self, tasklabel: str) -> None:
        self.tasklabel = tasklabel
        self.Mul = 0
        self.Add = 0


class VPU:
    def __init__(self) -> None:
        self.lanenum = MoraxConfig.LaneNum
        self.lanesize = MoraxConfig.LaneSize
        self.TimeFilm = TimeFilm()
        self.VPUActionList = []
        self.Busy = False

    def run_query(self, _q_vpu: QueryExcuteOnVPU, _issue_t: int):
        q_vpu = copy.deepcopy(_q_vpu)
        runtime = 0
        vts = TimeStamp(q_vpu.execution, _issue_t, q_vpu.tasklabel)
        vad = VPUActionDict(q_vpu.tasklabel)
        if q_vpu.dfmod == "PostProcess":  # postprocess of NVTC or TC
            (rowparts, collines) = q_vpu.tasksize
            runtime = rowparts * math.ceil(
                float(collines) / (self.lanesize * self.lanenum)
            )
            vad.Mul = 0
            vad.Add = (rowparts - 1) * collines
        elif q_vpu.dfmod == "SoftMAX":
            # NLT.Softmax1D 2D
            # max trick + [max-8） drop + exp LUT  + div LUT
            vdim = (
                q_vpu.layerclass.v_dim
                if isinstance(q_vpu.layerclass, Softmax1D)
                else q_vpu.layerclass.col_dim
            )
            # SO.VMAX
            if q_vpu.execution == VPUExe[7]:
                runtime = math.ceil((float(vdim) / self.lanesize) / self.lanenum) * (
                    self.lanesize ** 0.5
                )
                vad.Mul = 0
                vad.Add = get_reducetime(vdim)
            # SO.Vnorm
            elif q_vpu.execution == VPUExe[8]:
                runtime = runtime = (
                    math.ceil((float(vdim) / self.lanesize) / self.lanenum) * 2
                )
                vad.Mul = vdim
                vad.Add = vdim
        else:
            # LLT.VDP: one op
            # LLT.VMM == LLT.Linear: one output
            if q_vpu.execution == VPUExe[0] or q_vpu.execution == VPUExe[3]:
                vvtimes = math.ceil(
                    (float(q_vpu.layerclass.v_dim) / self.lanesize) / self.lanenum
                )
                runtime = vvtimes * (self.lanesize ** 0.5) + vvtimes
                vad.Mul = q_vpu.layerclass.v_dim
                vad.Add = get_reducetime(q_vpu.layerclass.v_dim)
            # LLT.VADD
            elif q_vpu.execution == VPUExe[1]:
                runtime = math.ceil(
                    (float(q_vpu.layerclass.v_dim) / self.lanesize) / self.lanenum
                )
                vad.Mul = 0
                vad.Add = q_vpu.layerclass.v_dim
            # LLT.VMUL
            elif q_vpu.execution == VPUExe[2]:
                runtime = math.ceil(
                    (float(q_vpu.layerclass.v_dim) / self.lanesize) / self.lanenum
                )
                vad.Mul = q_vpu.layerclass.v_dim
                vad.Add = 0
            # LLT.Batchnorm one channel of all batch, deprecated
            # LLT.Layernorm one batch
            elif q_vpu.execution == VPUExe[4] or q_vpu.execution == VPUExe[5]:
                R = math.ceil(float(q_vpu.layerclass.row_dim) / self.lanenum)
                C = math.ceil(float(q_vpu.layerclass.col_dim) / self.lanesize)
                runtime = R * C * 2
                vad.Mul = q_vpu.layerclass.row_dim * q_vpu.layerclass.col_dim
                vad.Add = q_vpu.layerclass.row_dim * q_vpu.layerclass.col_dim
            # NLT.Pooling one channel
            elif q_vpu.execution == VPUExe[6]:
                osize = q_vpu.layerclass.feature_size / q_vpu.layerclass.kernel_size
                runtime = (math.ceil(float(osize) / self.lanesize) / self.lanenum) ** 2
                vad.Mul = 0
                vad.Add = (q_vpu.layerclass.kernel_size - 1) * osize
        vts.update_span(runtime)
        self.TimeFilm.append_stamp(vts)
        self.VPUActionList.append(vad)
        return self.TimeFilm[-1].submit_t


def get_reducetime(_dim: int):
    add = 0
    while _dim > 1:
        add += _dim / 2
        _dim = _dim / 2
    return add
