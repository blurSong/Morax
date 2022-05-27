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
from morax.system.memonitor import Scratchpad

"""
# [bulk]
# indicate the data form of input and weight
# bulkfrom = (part or whole) FTR: BCHW    WET: KCRS    VEC:BM   MAT: BMN
# dataname = FTR WET VEC MAT
# bulklabel = modelname_ 'L'+layeridx_ 'WET_' bulksizeByte_ bulkfrom
# bulklabel = modelname_ 'L'+layeridx_ 'FTR_' bulksizeByte_ bulkfrom

            bulklabel: ResNet18_L17_WET_1024_K5_C50_RS0   ResNet18_L17_FTR_4096_B3_C7_HW3
#
    modelname: str
    layerindex: int
    datatype: str (WET or FTR or VEC or MAT)
    bulksizeByte: int
    bulkscratch: tuple(begin, end) or int
    K:  outc    B:          B:           B:
    C:  inc     C:          M:           M:
    RS:         H:                       N:
                W:
    bulklabel: str


# token: out-degree for F, total read demand for W.
"""


def make_bulklabel(mn, li, bsz, bsd: dict, datatype: str):
    assert datatype in ["WET", "FTR", "VEC", "MAT"]
    bulkform = str()
    for key, tup in bsd.item():
        """
        if isinstance(val, list):
            bulkform += "_" + key + "_".join(val)
        else:
            if val == 114514:
                bulkform += "_" + key + "ALL"
            else:
                bulkform += "_" + key + str(val)
        """
        if isinstance(tup, tuple):
            bulkform += "_" + key + str(tup[0]) + "-" + str(tup[1])
        else:
            bulkform += "_" + key + str(tup)
    return mn + "_L" + str(li) + "_" + datatype + "_" + str(bsz) + bulkform


class DataBulk:
    def __init__(
        self,
        _modelname,
        _layerindex,
        _datatype,
        _bulksizebyte,
        _bulkscratch: dict,
        _bulktoken=1,
    ) -> None:
        self.modelname = _modelname
        self.layerindex = _layerindex
        self.datatype = _datatype
        self.bulksizebyte = _bulksizebyte
        self.bulkscratch = copy.deepcopy(_bulkscratch)
        self.bulklabel = make_bulklabel(
            self.modelname,
            self.layerindex,
            self.bulksizebyte,
            self.bulkscratch,
            self.datatype,
        )
        self.token = _bulktoken


class BufferIOActionDict:
    def __init__(self, bulklabel: str) -> None:
        self.tasklbulklabelabel = bulklabel
        self.Read = 0
        self.Write = 0


class ScratchpadBuffer:
    def __init__(self, _sizeKB, _bandwidthgbps=0) -> None:
        self.CapacityByte = _sizeKB * 1024
        self.WaterLineByte = 0
        self.BandwidthGbps = _bandwidthgbps
        self.Scratchpad = Scratchpad()
        self.TimeFilm = TimeFilm()
        self.BufferIOList = []

    def write_buffer(self, _databulk: DataBulk):
        if self.WaterLineByte + _databulk.sizebyte > self.CapacityByte:
            # TODO
            raise Exception("Buffer overflowed.")
        elif _databulk.label in self.Scratchpad:
            # TODO
            raise Exception("Buffer duplicated.")
        else:
            # self.LabelSet.add(_databulk.label)
            self.Scratchpad.writeANote(_databulk)
            self.WaterLineByte += _databulk.sizebyte

    def read_buffer(self, _databulk: DataBulk) -> str:
        return self.Scratchpad.readANote(_databulk)

    def release(self, _note):
        self.WaterLineByte -= self.Scratchpad.get_size(_note)
        self.Scratchpad.delANote(_note)

    def merge_buffer(self, _note, _layerdict):
        self.Scratchpad.merge_scratchpad(_note, _layerdict)

    def run_query(self, _q_buffer: QueryBuffer, _issue_t) -> int:
        execution = _q_buffer.execution
        assert execution in [BO.Read, BO.Write]
        timestamplabel = "read_" if execution == BO.Read else "write_"
        biots = TimeStamp(
            _q_buffer.execution,
            _issue_t,
            timestamplabel + _q_buffer.databulkclass.bulklabel,
        )
        bioatd = BufferIOActionDict(_q_buffer.databulkclass.bulklabel)
        if _q_buffer.databulkclass.bulksizebyte > 0:
            if execution == BO.Write:
                self.write_buffer(_q_buffer.databulkclass)
                bioatd.Write = _q_buffer.databulkclass.sizebyte
            else:
                read_result = self.read_buffer(_q_buffer.databulkclass)
                if read_result == "Success":
                    bioatd.Read = _q_buffer.databulkclass.sizebyte
                else:
                    return -1
                    # need inter cluster read or dram read
        runtime = (
            _q_buffer.databulkclass.sizebyte * 8 / self.BandwidthGbps
            if self.BandwidthGbps != 0
            else 0
        )
        biots.update_span(runtime)
        self.TimeFilm.append_stamp_bufferver(biots)
        self.BufferIOList.append(bioatd)
        return biots.submit_t
        # return self.TimeFilm[-1].submit_t
