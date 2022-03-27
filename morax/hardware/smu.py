# SMU class of Morax
# Morax means the mixing of CMOS and RRAM
# Author: Tao Song.
# Last modification: 0327

from types import NoneType
import numpy as np
import subprocess as SP
import multiprocessing as MP
import copy
import math
from morax.system.interface import MoraxExecutionDict, ClusterComponent
from morax.system.timefilm import TimeFilm, TimeStamp
from collections import UserDict, UserList
from morax.system.config import MoraxConfig
from morax.system.query import QueryExcuteOnSMU
from morax.model.layer import Softmax1D, Softmax2D

SMUExe = MoraxExecutionDict[ClusterComponent.SMU]


class SMUActionDict:
    def __init__(self, _tasklabel: str) -> None:
        self.tasklabel = _tasklabel
        self.SO = -1
        self.Act = 0


class SMU:
    def __init__(self) -> None:
        self.TimeFilm = TimeFilm()
        self.SMUActionList = []
        self.regfile = (
            MoraxConfig.SMURegFileSizekB * 1024 * 8 / MoraxConfig.PrecisionBits
        )
        self.Busy = False

    def run_query(self, _q_smu: QueryExcuteOnSMU, issue_t: int):
        q_smu = copy.deepcopy(_q_smu)
        smts = TimeStamp(q_smu.execution, issue_t, q_smu.tasklabel)
        smad = SMUActionDict(q_smu.tasklabel)
        smad.SO = q_smu.execution
        runtime = 0
        # SO.Transpose
        if q_smu.execution == SMUExe[0]:
            M = q_smu.tasksize[0]
            N = q_smu.tasksize[1]
            runtime = N * M / self.regfile
            smad.Act = M * N
        # SO.Truncation
        if q_smu.execution == SMUExe[1]:
            V = (
                q_smu.layerclass.v_dim
                if isinstance(q_smu.layerclass, Softmax1D)
                else q_smu.layerclass.col_dim
            )
            runtime = V / self.regfile
            smad.Act = V
        # SO.HWNC2CHWN SO.CHWN2HWNC
        assert q_smu.dfmod == "UpStream"
        if q_smu.execution == SMUExe[2] or q_smu.execution == SMUExe[3]:
            M = q_smu.tasksize[0]
            N = q_smu.tasksize[1]
            runtime = N * M / self.regfile
            smad.Act = M * N
        smts.update_span(runtime)
        self.TimeFilm.append_stamp(smts)
        self.SMUActionList.append(smad)
        return smts.submit_t
