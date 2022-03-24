# TensorCore class of Morax
# Morax means the mixing of CMOS and RRAM
# Author: Tao Song.
# Last modification: 0324

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
from morax.system.query import QueryExcuteOnTC

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
        self.localbufsize = _bufsize
        self.rowbufsize = _pesize
        self.TimeFilm = TimeFilm()
        self.PEArrayActionList = []
        self.busy = False

    def run_subquery(self, _q_pearray, _issue_t: int, _layerclass):
        # _q_pearray: dfmod: str, execution: Enum, subtasksize: tuple(int, int) subtasklabel: str
        # subtasksize is the size of execution of this subquery on this PEArray
        assert (
            _q_pearray["subtasksize"][0] <= self.pesize
            and _q_pearray["subtasksize"][0] >= 0
            and _q_pearray["subtasksize"][1] <= self.pesize
            and _q_pearray["subtasksize"][1] >= 0
        )
        runtime = 0
        ts = TimeStamp(_q_pearray["execution"], _issue_t, _q_pearray["subtasklabel"])
        ad = PEArrayActionDict(_q_pearray["subtasklabel"])
        # LLT.Linear = LLT.VMM
        if _q_pearray["execution"] == TCExe[0] or _q_pearray["execution"] == TCExe[7]:
            assert _q_pearray["dfmod"] == "Systolic"
            M = _q_pearray["subtasksize"][0]
            N = _q_pearray["subtasksize"][1]
            ad["MAC"] = M * N
            ad["LBRead"] = M + M * N
            ad["LBWrite"] = N
            ad["RBActivate"] = N
            runtime = 1 + M + N + 1
        # LLT.CONV
        elif _q_pearray["execution"] == TCExe[1]:
            assert _q_pearray["dfmod"] == "OS"  # todo: for now, no CS
            M = _q_pearray["subtasksize"][0]
            N = _q_pearray["subtasksize"][1]
            K = _layerclass.kernel_size
            C = _layerclass.in_channel
            ad["MAC"] = M * N * K * K * C
            ad["LBRead"] = K * K * C + (N + K - 1) * (M + K - 1) * C
            ad["LBWrite"] = M * N
            ad["RBActivate"] = N * K * C
            runtime = (min(N, M) - 1 + K * K) * C
        # LLT.DWCONV
        elif _q_pearray["execution"] == TCExe[2]:
            M = _q_pearray["subtasksize"][0]
            N = _q_pearray["subtasksize"][1]
            K = _layerclass.kernel_size
            C = _layerclass.channel
            ad["MAC"] = M * N * K * K
            ad["LBRead"] = K * K + (N + K - 1) * (M + K - 1)
            ad["LBWrite"] = M * N
            ad["RBActivate"] = N * K
            runtime = min(N, M) - 1 + K * K
        # LLT.Residual
        elif _q_pearray["execution"] == TCExe[3]:
            M = _q_pearray["subtasksize"][0]
            N = _q_pearray["subtasksize"][1]
            C = _layerclass.channel
            ad["MAC"] = M * N
            ad["LBRead"] = M * N * 2
            ad["LBWrite"] = M * N
            ad["RBActivate"] = 0
            runtime = max(N, M)
        # LLT.Batchnorm
        elif _q_pearray["execution"] == TCExe[4]:
            M = _q_pearray["subtasksize"][0]
            N = _q_pearray["subtasksize"][1]
            ad["MAC"] = M * N * 2
            ad["LBRead"] = M * N
            ad["LBWrite"] = N * N
            ad["RBActivate"] = 0
            runtime = max(N, M) + 1
        # LLT.TRCONV
        elif _q_pearray["execution"] == TCExe[5]:
            M = _q_pearray["subtasksize"][0]
            N = _q_pearray["subtasksize"][1]
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
            M = _q_pearray["subtasksize"][0]
            N = _q_pearray["subtasksize"][1]
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
            M = _q_pearray["subtasksize"][0]
            N = _q_pearray["subtasksize"][1]
            KDIM = _layerclass.k_dim
            ad["MAC"] = M * KDIM * N
            ad["LBRead"] = N * KDIM + M * KDIM
            ad["LBWrite"] = M * N
            ad["RBActivate"] = N * KDIM
            runtime = 1 + KDIM + M
        # LLT.MADD
        elif _q_pearray["execution"] == TCExe[9]:
            M = _q_pearray["subtasksize"][0]
            N = _q_pearray["subtasksize"][1]
            ad["MAC"] = M * N
            ad["LBRead"] = M * N * 2
            ad["LBWrite"] = M * N
            ad["RBActivate"] = 0
            runtime = max(M, N)
        # LLT.Layernorm
        elif _q_pearray["execution"] == TCExe[10]:
            M = _q_pearray["subtasksize"][0]
            N = _q_pearray["subtasksize"][1]
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
        return ts.submit_t  # no need to push _t to backward


class TensorCoreNOC:
    def __init__(self, _pearraynum, _nocbw) -> None:
        self.fanoutbus = _pearraynum
        self.nocbw = _nocbw
        self.NOCCastList = []
        # self.TimeFilm = TimeFilm()

    def run_query(self, _q_noc, _issue_t: int = 0, _bulksize: float = 0):
        # _q_noc: tasksizelist: list of tuple(int, int) tasklabel: str
        # _bulksize is the total databulk to issue of this query
        tslist = []
        ncl = NOCCastList(_q_noc["tasklabel"])
        for fob in range(self.fanoutbus):
            (ts0, ts1) = _q_noc["tasksizelist"][fob]
            assert ts0 >= 0 and ts1 >= 0
            tsf = ts0 * ts1
            tslist.append(tsf)
        tssum = sum(tslist)
        for fob in range(self.fanoutbus):
            bs = _bulksize * tslist[fob] / tssum
            ncl.append(bs)
        self.NOCCastList.append(ncl)
        # add NOC Cast time to update submit_t
        submit_t = _issue_t + max(ncl) / self.nocbw
        return submit_t


# Morax DONOT ALLOW asynchronous execution within one TC


class TensorCore:
    def __init__(
        self,
        _tcid: int,
        _pearraynum: int,
        _pearraysize: int,
        _pearraybufsize: int,
        _nocbw: int,
    ) -> None:
        self.tcid = _tcid
        self.pearraynum = _pearraynum
        self.PEArrayObjList = []
        for peid in range(self.pearraynum):
            pearray = PEArray(peid, _pearraysize, _pearraybufsize)
            self.PEArrayObjList.append(copy.deepcopy(pearray))
        self.NOC = TensorCoreNOC(self.pearraynum, _nocbw)
        self.TimeFilm = TimeFilm()

    def run_query(
        self, _qclass_tc: QueryExcuteOnTC, _issue_t: int = 0, _bulksize: int = 0
    ) -> int:
        """_qclass_tc:
        class QueryExcuteOnTC(QueryExcute)
        # layerclass,
        # tasklabel: str,
        # dfmod: str,
        # execution: Enum
        # tasksizelist: list(tuple(int, int))
        """
        # 1. parse query
        query_tc = copy.deepcopy(_qclass_tc)
        sq_pearraylist = []
        receive_t_list = []
        sq_pearray = {}
        assert len(query_tc.tasksizelist) == self.pearraynum
        for peid in range(self.pearraynum):
            sq_pearray["subtasksize"] = query_tc.tasksizelist[peid]
            sq_pearray["dfmod"] = query_tc.dfmod
            sq_pearray["execution"] = query_tc.execution
            sq_pearray["subtasklabel"] = query_tc.tasklabel + "_pe" + str(peid)
            sq_pearraylist.append(copy.deepcopy(sq_pearray))
        # 2. justify invoke time
        q_noc = {"tasklabel": query_tc.tasklabel, "tasksizelist": query_tc.tasksizelist}
        invoke_t = self.NOC.run_query(q_noc, _issue_t, _bulksize)
        # 3. on-fly
        timestamp = TimeStamp(query_tc.execution, _issue_t, query_tc.tasklabel)
        for peid in range(self.pearraynum):
            if (
                sq_pearraylist[peid]["subtasksize"][0] > 0
                and sq_pearraylist[peid]["subtasksize"][1] > 0
            ):
                receive_t_list[peid] = self.PEArrayObjList[peid].run_subquery(
                    sq_pearraylist[peid], invoke_t, query_tc.layerclass
                )
            else:
                receive_t_list[peid] = invoke_t
        submit_t = max(receive_t_list)
        timestamp.update_submit_t(submit_t)
        self.TimeFilm.append_stamp(timestamp)
        return self.TimeFilm[-1].submit_t


def is_zero(size):
    return size > -0.0000000009 and size < 0.0000000009
