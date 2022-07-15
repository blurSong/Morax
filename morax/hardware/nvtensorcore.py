# nvTensorCore class of Morax
# Morax means the mixing of CMOS and RRAM
# Author: Tao Song.
# Last modification: 0324

import numpy as np
import subprocess as SP
import multiprocessing as MP
import copy
from morax.system.interface import MoraxExecutionDict, ClusterComponent
from morax.system.timefilm import TimeFilm, TimeStamp
from collections import UserDict
from morax.system.config import MoraxConfig, HWParam
from morax.system.query import QueryExcuteOnNVTC


nvTCExe = MoraxExecutionDict[ClusterComponent.nvTensorCore]


class RRAMSliceMVMActionDict:  # 1 Xbar
    def __init__(self) -> None:
        self.subtasklabel = "nolabel"
        self.DA = 0
        self.AD = 0
        self.Xbar = 0  # Total MAC NUM
        self.ShiftAdd = 0
        self.RegRead = 0
        self.MVMAcc = 0


class RRAMSliceLUTActionDict:  # 1 Slice = 8 Xbar = 1 ISAAC IMA / MNSIM PE
    def __init__(self, _subtasklabel) -> None:
        self.subtasklabel = _subtasklabel
        self.RegRead = 0
        self.LookUp = 0
        self.SenceAmp = 0


# NOTE RRAM is static so one nvtc has 100% run-time information
# TODO: remove ideal LUT setup

"""
class RRAMXbar:
    def __init__(
        self,
        # _xbarid,
        # _layerids: tuple(int, int),
        _mvmrow,
        _mvmcol,
        _islut=True,
    ) -> None:
        self.size = MoraxConfig.RRAMXbarSize
        # self.xbarid = _xbarid
        # self.layerid_tuple = _layerids  # (modelid, layerid)
        self.mvmrow = _mvmrow
        self.mvmcol = _mvmcol
        self.lutrow = MoraxConfig.RRAMLUTRows if _islut else 0
        self.mapped = False

    def map_xbar(self, _mvmrow, _mvmcol, _islut):
        self.mapped = True
        self.mvmrow = _mvmrow
        self.mvmcol = _mvmcol
        self.lutrow = MoraxConfig.RRAMLUTRows if _islut else 0
"""


class RRAMSlice:
    def __init__(
        self,
        _sliceid,
        _layerids: tuple = (0, 0),
        _islut=True,
        _mvmrow=MoraxConfig.RRAMXbarSize - MoraxConfig.RRAMLUTRows,
        _mvmcol=MoraxConfig.RRAMXbarSize,
    ) -> None:
        self.xbarnum = MoraxConfig.RRAMXbarNum
        self.xbarsize = MoraxConfig.RRAMXbarSize
        self.sliceid = _sliceid
        self.layerids = _layerids
        self.EightXbar = {
            "mvmrow": _mvmrow,
            "mvmcol": _mvmcol,
            "lutrow": MoraxConfig.RRAMLUTRows if _islut else 0,
        }
        self.RRAMSliceMVMAction = RRAMSliceMVMActionDict()
        self.RRAMSliceLUTActionList = []
        self.TimeFilm = TimeFilm()  # MVM once BUT LUT many times
        self.layerinfo = (-1, (0, 0))
        self.mapped = False

    # info: (modelname, layerid, mappingmap(rowid, colid))
    def map_slice(
        self, _layerinfo: tuple, _mvmrow, _mvmcol, _lutrow,
    ):
        self.mapped = True
        self.layerinfo = _layerinfo
        self.EightXbar["mvmrow"] = _mvmrow
        self.EightXbar["mvmcol"] = _mvmcol
        self.EightXbar["lutrow"] = _lutrow

    def run_query_static(self, _q_slice, _issue_t: int = 0) -> int:
        runtime = 0
        ts = TimeStamp(_q_slice["execution"], _issue_t, _q_slice["subtasklabel"])
        # LLT.ALL
        if _q_slice["execution"] in range(6):
            assert _q_slice["dfmod"] == "Xbar"
            self.RRAMSliceMVMAction.subtasklabel = _q_slice["subtasklabel"]
            self.RRAMSliceMVMAction.DA = self.xbarnum * self.EightXbar["mvmrow"]
            self.RRAMSliceMVMAction.AD = self.xbarnum * self.EightXbar["mvmcol"]
            self.RRAMSliceMVMAction.Xbar = (
                self.xbarnum * self.EightXbar["mvmrow"] * self.EightXbar["mvmcol"]
            )
            self.RRAMSliceMVMAction.ShiftAdd = self.xbarnum * MoraxConfig.PrecisionBits
            self.RRAMSliceMVMAction.RegRead = self.EightXbar["mvmrow"]
            self.RRAMSliceMVMAction.MVMAcc = self.EightXbar["mvmcol"]
            runtime = (
                int(self.EightXbar["mvmcol"] // HWParam.ADCSpeedGbps)
                * MoraxConfig.PrecisionBits
                + MoraxConfig.RRAMXbarNum
                + 1
            )
        # SO.LookUp
        elif _q_slice["execution"] == nvTCExe[6]:
            lut = RRAMSliceLUTActionDict(_q_slice["subtasklabel"])
            lookuptime = 2 if _q_slice["dfmod"] == "LUT16" else 1
            lutedxbarnum = (
                2 ** (3 - ((7 + MoraxConfig.RRAMLUTRows ** 0.5) - 8)) * lookuptime
            )
            lut.LookUp = lutedxbarnum
            lut.RegRead = lookuptime
            lut.SenceAmp = lutedxbarnum
            runtime = lookuptime
            self.RRAMSliceLUTActionList.append(lut)
        ts.update_span(runtime)
        self.TimeFilm.append_stamp(ts)
        return ts.submit_t


class RRAMSlicesTree:
    def __init__(self) -> None:
        if MoraxConfig.RRAMSlicesTreeType == 1:
            self.treetype = "2LeavesTree"
            self.hops = 2
        elif MoraxConfig.RRAMSlicesTreeType == 2:
            self.treetype = "HTree"
            self.hops = MoraxConfig.RRAMSliceNum ** 0.5 - 1
        self.TreeCastList = []

    def downstream(
        self, _issue_t: int, _q_tree=None,
    ):
        # _q_tree: tasksizelist: list of tuple(treenode_id, leaf_id) subtasklabel: str
        # _bulksize is the total databulk to issue of this query
        return _issue_t + self.hops

    def uptream(
        self, _issue_t: int, _q_tree=None,
    ):
        # _q_tree: tasksizelist: list of tuple(treenode_id, leaf_id) subtasklabel: str
        # _bulksize is the total databulk to issue of this query
        return _issue_t + self.hops


class nvTensorCore:
    def __init__(self, _nvtcid: int) -> None:
        self.nvtcid = _nvtcid
        self.slicenum = MoraxConfig.RRAMSliceNum
        self.RRAMSliceObjList = []
        for sliceid in range(self.slicenum):
            slice = RRAMSlice(sliceid)
            self.RRAMSliceObjList.append(copy.deepcopy(slice))
        self.Tree = RRAMSlicesTree()
        self.TimeFilm = TimeFilm()

    def map_slices(  # depracated
        self, _sliceid_list, _layerinfotuple_list, _islut_list, _mvmsizetuple_list
    ):
        assert len(_sliceid_list) <= self.slicenum
        for sid in _sliceid_list:
            self.RRAMSliceObjList[sid].map_slice(
                _layerinfotuple_list[sid],
                _islut_list[sid],
                _mvmsizetuple_list[sid][0],
                _mvmsizetuple_list[sid][1],
            )

    def map_a_slice(self, _sliceid, _mapping_info, _mvm_row, _mvm_col, _lut_row):
        self.RRAMSliceObjList[_sliceid].map_slice(
            _mapping_info, _mvm_row, _mvm_col, _lut_row
        )

    def run_query(self, _qclass_nvtc: QueryExcuteOnNVTC, _issue_t: int):
        query_nvtc = copy.deepcopy(_qclass_nvtc)
        q_slice_list = []
        receive_t_list = []
        q_slice = {}
        for sliceid in query_nvtc.sliceidlist:
            q_slice["dfmod"] = query_nvtc.dfmod
            q_slice["subtasklabel"] = (
                query_nvtc.tasklabel
                + "_nvtc"
                + str(self.nvtcid)
                + "_slice"
                + str(sliceid)
            )
            q_slice["execution"] = query_nvtc.execution
            q_slice_list.append(copy.deepcopy(q_slice))
        issue_t = _issue_t + self.Tree.downstream()
        timestamp = TimeStamp(query_nvtc.execution, _issue_t, query_nvtc.tasklabel)
        for sliceid in query_nvtc.sliceidlist:
            receive_t_list.append(
                self.RRAMSliceObjList[sliceid].run_query_static(
                    q_slice_list.pop(0), issue_t
                )
            )
        submit_t = max(receive_t_list) + self.Tree.uptream()
        timestamp.update_submit_t(submit_t)
        self.TimeFilm.append_stamp(timestamp)
        return self.TimeFilm[-1].submit_t

