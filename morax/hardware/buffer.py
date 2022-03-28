# ScratchPad Buffer class of Morax
# Morax means the mixing of CMOS and RRAM
# Author: Tao Song.
# Last modification: 0327

import numpy as np
import subprocess as SP
import multiprocessing as MP
import copy
from morax.system.interface import BO, ClusterComponent
from morax.system.timefilm import TimeFilm, TimeStamp
from collections import UserDict
from morax.system.config import MoraxConfig, HWParam
from morax.system.query import QueryBuffer

"""
# [bulk]
# indicate the data form of input and weight
# bulkfrom = (part or whole) feature: BCHW  weight: KCRS  MVM & GEMM: BMKN
# dataname = W or F
# bulklabel = modelname_ 'L'+layeridx_                'WET'+ bulkidx_ bulksizeByte_ bulkfrom
# bulklabel = modelname_ 'L'+layeridx_ 'B'+ batchidx_ 'FTR'+ bulkidx_ bulksizeByte_ bulkfrom

            bulklabel: ResNet18_L17_WET5_2304_CRS   ResNet18_L17_B3_FTR2_4096_HW

# token: out-degree for F, total read demand for W.
"""


class DataBulk:
    def __init__(self, _bulksizebyte, _bulklabel: str, _bulktoken=1) -> None:
        self.sizebyte = _bulksizebyte * 1024
        self.label = _bulklabel
        self.token = _bulktoken


class BufferIOActionDict:
    def __init__(self, bulklabel: str) -> None:
        self.tasklbulklabelabel = bulklabel
        self.Read = 0
        self.Write = 0


class ScratchPadBuffer:
    def __init__(self, _sizekb, _bandwidthgbps=0) -> None:
        self.CapacityByte = _sizekb * 1024
        self.WaterLineByte = 0
        self.BandwidthGbps = _bandwidthgbps
        self.MemScratchPad = {}
        # self.LabelSet = set()
        self.TimeFilm = TimeFilm()
        self.BufferIOList = []

    def write_buffer(self, _databulk: DataBulk):
        if self.WaterLineByte + _databulk.sizebyte > self.CapacityByte:
            # TODO
            raise Exception("Buffer overflowed.")
        elif _databulk.label in self.MemScratchPad:
            # TODO
            raise Exception("Buffer duplicated.")
        else:
            # self.LabelSet.add(_databulk.label)
            self.MemScratchPad[_databulk.label] = copy.deepcopy(_databulk)
            self.WaterLineByte += _databulk.sizebyte

    def read_buffer(self, _datalabel: str) -> str:
        if _datalabel in self.MemScratchPad:
            self.MemScratchPad[_datalabel].token -= 1
            if self.MemScratchPad[_datalabel].token == 0:
                self.WaterLineByte -= self.MemScratchPad[_datalabel].sizebyte
                del self.MemScratchPad[_datalabel]
            return "Success"
        else:
            # Shuld read form other cluster or DRAM
            return "False"

    def run_query(self, _q_buffer: QueryBuffer, _issue_t) -> int:
        execution = _q_buffer.execution
        assert execution in [BO.Read, BO.Write]
        biots = TimeStamp(_q_buffer.execution, _issue_t, _q_buffer.databulkclass.label)
        bioatd = BufferIOActionDict(_q_buffer.databulkclass.label)
        if execution == BO.Write:
            self.write_buffer(_q_buffer.databulkclass)
            bioatd.Write = _q_buffer.databulkclass.sizebyte
        else:
            read_result = self.read_buffer(_q_buffer.databulkclass.label)
            if read_result == "Success":
                bioatd.Read = _q_buffer.databulkclass.sizebyte
            else:
                return -1
                # need inter cluster read or dram read
        runtime = (
            _q_buffer.databulkclass.sizebyte
            * MoraxConfig.PrecisionBits
            / self.BandwidthGbps
            if self.BandwidthGbps != 0
            else 0
        )
        biots.update_span(runtime)
        self.TimeFilm.append_stamp(biots)
        self.BufferIOList.append(bioatd)
        return biots.submit_t
