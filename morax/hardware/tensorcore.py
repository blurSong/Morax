# TensorCore class of Morax
# Morax means the mixing of CMOS and RRAM
# Author: Tao Song.
# Last modification: 0316

import sys
from typing import Container
import re
import numpy as np
import subprocess as SP
import multiprocessing as MP
import openpyxl
import math
import copy
from morax.system.interface import MoraxExecutionDict, ClusterComponent
from morax.model.layer import LinearLayerType as LLT, NonlinearLayerType as NLT
from morax.system.timefilm import TimeFilm, TimeStamp
from typing import Dict, List

TCExe = MoraxExecutionDict[ClusterComponent.TensorCore]


class PEArrayActionDict(Dict):
    def __init__(self, _subtasklabel) -> None:
        super().__init__()
        self.subtasklabel = _subtasklabel
        self["MAC"] = 0
        self["LBRead"] = 0
        self["LBWrite"] = 0
        self["RBActivate"] = 0


class NOCCastList(List):
    def __init__(self, _tasklabel) -> None:
        super().__init__()
        self.tasklabel = _tasklabel


class PEArray:
    def __init__(self, _peid: int, _pesize: int, _bufsize: int) -> None:
        self.peid = _peid
        self.pesize = _pesize
        self.localbuffer_size = _bufsize
        self.rowbuf_size = _pesize
        self.TimeFilm = TimeFilm()
        self.PEArrayActionList = []
        self.busy = False

    def run_subquery(self, _q_pearray, _issue_t: int, _layerclass):
        # q_pearray: dfmod: str, execution: Enum, subtasksize: tuple(float, float) subtasklabel: str
        # subtasksize is the size of execution of this subquery on this PEArray
        assert (
            _q_pearray["subtasksize"][0] <= 1.000000000009
            and _q_pearray["subtasksize"][0] > 0
        )
        runtime = 0
        ts = TimeStamp(_q_pearray["execution"], _issue_t, _q_pearray["subtasklabel"])
        ad = PEArrayActionDict(_q_pearray["subtasklabel"])
        # LLT.Linear = LLT.VMM
        if _q_pearray["execution"] == TCExe[0] or _q_pearray["execution"] == TCExe[7]:
            assert _q_pearray["dfmod"] == "Systolic"
            M = int(self.pesize * _q_pearray["subtasksize"][0])
            N = int(self.pesize * _q_pearray["subtasksize"][1])
            ad["MAC"] = M * N
            ad["LBRead"] = M + M * N
            ad["LBWrite"] = N
            ad["RBActivate"] = N
            runtime = 1 + M + N + 1
        # LLT.CONV
        elif _q_pearray["execution"] == TCExe[1]:
            assert _q_pearray["dfmod"] == "OS"  # todo: for now, no CS
            M = int(self.pesize * _q_pearray["subtasksize"][0])
            N = int(self.pesize * _q_pearray["subtasksize"][1])
            K = _layerclass.kernel_size
            C = _layerclass.in_channel
            ad["MAC"] = M * N * K * K * C
            ad["LBRead"] = K * K * C + (N + K - 1) * (M + K - 1) * C
            ad["LBWrite"] = M * N
            ad["RBActivate"] = N * K * C
            runtime = (min(N, M) - 1 + K * K) * C
        # LLT.DWCONV
        elif _q_pearray["execution"] == TCExe[2]:
            M = int(self.pesize * _q_pearray["subtasksize"][0])
            N = int(self.pesize * _q_pearray["subtasksize"][1])
            K = _layerclass.kernel_size
            C = _layerclass.channel
            ad["MAC"] = M * N * K * K
            ad["LBRead"] = K * K + (N + K - 1) * (M + K - 1)
            ad["LBWrite"] = M * N
            ad["RBActivate"] = N * K
            runtime = min(N, M) - 1 + K * K
        # LLT.Residual
        elif _q_pearray["execution"] == TCExe[3]:
            M = int(self.pesize * _q_pearray["subtasksize"][0])
            N = int(self.pesize * _q_pearray["subtasksize"][1])
            C = _layerclass.channel
            ad["MAC"] = M * N
            ad["LBRead"] = M * N * 2
            ad["LBWrite"] = M * N
            ad["RBActivate"] = 0
            runtime = max(N, M)
        # LLT.Batchnorm
        elif _q_pearray["execution"] == TCExe[4]:
            M = int(self.pesize * _q_pearray["subtasksize"][0])
            N = int(self.pesize * _q_pearray["subtasksize"][1])
            ad["MAC"] = M * N * 2
            ad["LBRead"] = M * N
            ad["LBWrite"] = N * N
            ad["RBActivate"] = 0
            runtime = max(N, M) + 1
        # LLT.TRCONV
        elif _q_pearray["execution"] == TCExe[5]:
            M = int(self.pesize * _q_pearray["subtasksize"][0])
            N = int(self.pesize * _q_pearray["subtasksize"][1])
            KO = _layerclass.kernel_size
            KD = (KO + 1) * _layerclass.dilation - 1
            C = _layerclass.in_channel
            ad["MAC"] = M * N * KO * KO * C
            ad["LBRead"] = KO * KO * C + (N + KD - 1) * (M + KD - 1) * C
            ad["LBWrite"] = M * N
            ad["RBActivate"] = N * KD * C
            runtime = (min(N, M) - 1 + KD * KD) * C
        # LLT.NGCONV
        elif _q_pearray["execution"] == TCExe[6]:
            M = int(self.pesize * _q_pearray["subtasksize"][0])
            N = int(self.pesize * _q_pearray["subtasksize"][1])
            K = _layerclass.kernel_size
            CO = _layerclass.in_channel
            CP = CO / _layerclass.group
            ad["MAC"] = M * N * K * K * CP
            ad["LBRead"] = K * K * CP + (N + K - 1) * (M + K - 1) * CP
            ad["LBWrite"] = M * N
            ad["RBActivate"] = N * K * CP
            runtime = (min(N, M) - 1 + K * K) * CP
        # LLT.GEMM
        elif _q_pearray["execution"] == TCExe[8]:
            M = int(self.pesize * _q_pearray["subtasksize"][0])
            N = int(self.pesize * _q_pearray["subtasksize"][1])
            KDIM = _layerclass.k_dim
            ad["MAC"] = M * KDIM * N
            ad["LBRead"] = N * KDIM + M * KDIM
            ad["LBWrite"] = M * N
            ad["RBActivate"] = N * KDIM
            runtime = 1 + KDIM + M
        # LLT.MADD
        elif _q_pearray["execution"] == TCExe[9]:
            M = int(self.pesize * _q_pearray["subtasksize"][0])
            N = int(self.pesize * _q_pearray["subtasksize"][1])
            ad["MAC"] = M * N
            ad["LBRead"] = M * N * 2
            ad["LBWrite"] = M * N
            ad["RBActivate"] = 0
            runtime = max(M, N)
        # LLT.Layernorm
        elif _q_pearray["execution"] == TCExe[10]:
            M = int(self.pesize * _q_pearray["subtasksize"][0])
            N = int(self.pesize * _q_pearray["subtasksize"][1])
            ad["MAC"] = M * N * 2
            ad["LBRead"] = M * N
            ad["LBWrite"] = N * N
            ad["RBActivate"] = 0
            runtime = max(N, M) + 1
        # NLT.Pooling   # deprecated
        # elif _q_pearray['execution'] == TCExe[11]:
        ts.update_span(runtime)
        self.TimeFilm.append_stamp(ts)
        self.PEArrayActionList.append(ad)
        return ts.submit_t


class TensorCoreNOC:
    def __init__(self, _pearraynum) -> None:
        self.fanoutbus = _pearraynum
        self.NOCCastList = []

    def run_query(self, _q_noc, _issue_t: int = 0, _bulksize: float = 0):
        # q_noc: tasksize: list of tuple(float, float) tasklabel: str
        # _bulksize is the total databulk to issue of this query
        tslist = []
        ncd = NOCCastList(_q_noc["tasklabel"])
        for fob in range(self.fanoutbus):
            (ts0, ts1) = _q_noc["tasksize"][fob]
            assert ts0 > -0.0000009 and ts1 > -0.0000009
            tsf = ts0 * ts1
            tslist.append(tsf)
        tssum = sum(tslist)
        for fob in range(self.fanoutbus):
            bs = _bulksize * tslist[fob] / tssum
            ncd.append(bs)
        self.NOCCastList.append(ncd)
        # todo: add NOC Cast time to update submit_t
        return


class TensorCore:
    def __init__(
        self, _tcid: int, _pearraynum: int, _pearraysize: int, _pearraybufsize: int
    ) -> None:
        self.tcid = _tcid
        self.pearraynum = _pearraynum
        self.PEArrayObjList = []
        for pea in range(self.pearraynum):
            pearray = PEArray(pea, _pearraysize, _pearraybufsize)
            self.PEArrayObjList.append(copy.deepcopy(pearray))
        self.NOC = TensorCoreNOC(self.pearraynum)

    def run_query():
        return

