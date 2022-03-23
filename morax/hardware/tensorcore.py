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
from morax.system.interface import MoraxExecutionDict, ClusterComponent
from morax.model.layer import LinearLayerType as LLT, NonlinearLayerType as NLT
from morax.system.timefilm import TimeFilm, TimeStamp
from typing import Dict

int64 = np.int64

TCExe = MoraxExecutionDict[ClusterComponent.TensorCore]


class PEArrayActionDict(Dict):
    def __init__(self, _tasklabel) -> None:
        super().__init__()
        self.tasklabel = _tasklabel
        self['MAC'] = 0
        self['LBRead'] = 0
        self['LBWrite'] = 0
        self['RBActivate'] = 0



class PEArray():
    def __init__(self, _pesize, _bufsize) -> None:
        self.pesize = _pesize
        self.localbuffer_size = _bufsize
        self.rowbuf_size = _pesize
        self.TimeFilm = TimeFilm()
        self.PEArrayActionList = []
        self.busy = False

    def run_query(self, _q_pearray, _issue_t: int, _layerclass):
        # q_pearray: _dfmod: str, _execution: Enum, _subtasksize: tuple(float, float) tasklabel: str
        # _subtasksize is the size of execution of this subquery on this PEArray
        assert _q_pearray['subtasksize'] <= 1.0 and _q_pearray['subtasksize'] > 0
        runtime = 0
        ts = TimeStamp(_q_pearray['execution'], _issue_t, _q_pearray['tasklabel'])
        ad = PEArrayActionDict(_q_pearray['tasklabel'])
        # LLT.Linear
        if _q_pearray['execution'] == TCExe[0]:
            assert _q_pearray['dfmod'] == 'Systolic'
            R = int(self.pesize * _q_pearray['subtasksize'][0])
            C = int(self.pesize * _q_pearray['subtasksize'][1])
            ad['MAC'] = R * C
            ad['LBRead'] = R + R * C
            ad['LBWrite'] = C
            ad['RBActivate'] = C
            runtime = 1 + R + C + 1
        # LLT.CONV
        elif _q_pearray['execution'] == TCExe[1]:
            assert _q_pearray['dfmod'] == 'OS'  # todo: for now, no CS
            N = int(self.pesize * _q_pearray['subtasksize'][0])
            K = _layerclass.kernel_size
            C = _layerclass.in_channel
            ad['MAC'] = N * N * K * K * C
            ad['LBRead'] = K * K * C + (N + K - 1) ** 2 * C
            ad['LBWrite'] = N * N
            ad['RBActivate'] = (N + K - 1) * K
            runtime = (N - 1 + K * K) * C
        # LLT.DWCONV
        elif _q_pearray['execution'] == TCExe[2]:
            N = int(self.pesize * _q_pearray['subtasksize'][0])
            K = _layerclass.kernel_size
            C = _layerclass.channel
            ad['MAC'] = N * N * K * K 
            ad['LBRead'] = K * K + (N + K - 1) ** 2
            ad['LBWrite'] = N * N
            ad['RBActivate'] = (N + K - 1) * K
            runtime = N - 1 + K * K
        # LLT.Residual
        elif _q_pearray['execution'] == TCExe[3]:
            N = int(self.pesize * _q_pearray['subtasksize'][0])
            K = _layerclass.kernel_size
            C = _layerclass.channel
            ad['MAC'] = N * N * K * K 
            ad['LBRead'] = K * K + (N + K - 1) ** 2
            ad['LBWrite'] = N * N
            ad['RBActivate'] = (N + K - 1) * K
            runtime = N - 1 + K * K
        ts.update_span(runtime)
        self.TimeFilm.append_stamp(ts)
        self.PEArrayActionList.append(ad)





class TensorCore():


