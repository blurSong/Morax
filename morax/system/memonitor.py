from morax.hardware.buffer import DataBulk
from morax.system.config import MoraxConfig, HWParam
from morax.system.query import QueryRingBus, QueryDMA
import copy
from morax.system.interface import *


def edit_data_index(_thisidx, _thatidx, datatype):
    # for concat or append
    return


def check_scratchpad():
    return


def takeone(_tuptup: tuple):
    return _tuptup[0][0]


def check_range(c0, c1, _tupc):
    if c1 < _tupc[0] or c0 > _tupc[1]:
        return -1, -1
    else:
        c00 = c0 if c0 >= _tupc[0] else _tupc[0]
        c11 = c1 if c1 <= _tupc[1] else _tupc[1]
        return c00, c11


""" a scratchpad note is:
|-------------------------------------|
| note                                |
| [modelname + layerindex + datatype] |
|-------------------------------------|
| sizebyte                            |              
| # token                             |
|-------------------------------------|
| bulklabel1 | sizebyte | scratchdict |
| bulklabel2 | sizebyte | scratchdict |
| bulklabel3 | sizebyte | scratchdict |
|-------------------------------------|
"""


class bulknote:
    def __init__(self, _databulk: DataBulk) -> None:
        self.bulklabel = _databulk.bulklabel
        self.sizebyte = _databulk.bulksizebyte
        self.scratchdict = _databulk.bulkscratch


class Scratchpad:
    # 1 改concat的index
    # 2 改卷积和矩阵乘的数据标签
    def __init__(self):
        self.Scratchpad = {}
        return

    def writeANote(self, _bulk: DataBulk):
        note = _bulk.modelname + "_" + str(_bulk.layerindex) + "_" + _bulk.datatype
        if note in self.Scratchpad:
            self.Scratchpad[note]["sizebyte"] += _bulk.bulksizebyte
            abulknote = bulknote(_bulk)
            if abulknote
            self.Scratchpad[note]["bulknotelist"].append(copy.deepcopy(abulknote))
            # assert self.Scratchpad[note]["token"] == _bulk.token
            # NOTE token is maintained in monitor
        else:
            pad = {}
            pad["sizebyte"] = _bulk.bulksizebyte
            abulknote = bulknote(_bulk)
            pad["bulknotelist"] = [abulknote]
            # pad["token"] = _bulk.token
            self.Scratchpad[note] = copy.deepcopy(pad)

    def check_scratchpad(self, _bulk: DataBulk):
        # return subbulk size only
        note = _bulk.modelname + "_" + str(_bulk.layerindex) + "_" + _bulk.datatype
        if note not in self.Scratchpad:
            return 0
        else:
            if _bulk.datatype == "WET":
                klist = []
                clist = []
                for bulknote in self.Scratchpad[note]["bulknotelist"]:
                    klist.append(bulknote.scratchdict["K"])
                    clist.append(bulknote.scratchdict["C"])
                kclist = list(zip(klist, clist))  # [((k0,k1), (c0,c1)), ..., ]
                kclist.sort(key=takeone, reverse=False)
                k0 = _bulk.bulkscratch["K"][0]
                k1 = _bulk.bulkscratch["K"][1]
                c0 = _bulk.bulkscratch["C"][0]
                c1 = _bulk.bulkscratch["C"][1]
                [c00, c11, k00, k11] = [-1] * 4
                subbulksize = 0
                for kctup in kclist:
                    if kctup[0][0] > k1:
                        break
                    if kctup[0][1] < k0:
                        continue
                    if kctup[1][0] > c1 or kctup[1][1] < c0:
                        continue
                    else:
                        c00, c11 = check_range(c0, c1, kctup[1])
                        k00, k11 = check_range(k0, k1, kctup[0])
                        subbulksize += _bulk.bulksizebyte * (
                            (c11 - c00 + 1)
                            * (k11 - k00 + 1)
                            * 1.0
                            / ((k1 - k0 + 1) * (c1 - c0 + 1))
                        )
            return subbulksize

    def readANote(self, _bulk: DataBulk):
        if check_scratchpad(_bulk) > 0:
            return "Success"
        else:
            return "Fail"

    def delANote(self, _note):
        assert _note in self.Scratchpad
        del self.Scratchpad[_note]

    def clearScratchpad(self):
        self.Scratchpad.clear()


#  a cheat-sheet to record the locations and token of mm_idx_type
class Memonitor:
    def __init__(self) -> None:
        self.scratchpadnum = MoraxConfig.ClusterNum
        self.monitor = {}

    def insert_note(self, _note: str, _location: int):
        """
        if _note not in self.monitor:
            self.monitor[_note]["loclist"] = [_location]
            # self.monitor[_note]["token"] = _token
            assert _location in range(self.scratchpadnum)  # -1 for offchip (depr)
        else:
            # assert _token == self.monitor[_note]["token"]
        """
        self.monitor[_note]["loclist"].append(_location)

    def transfer_note(self, _note, _from, _to):
        assert _from in self.monitor[_note]["loclist"]
        if _to not in self.monitor[_note]["loclist"]:
            self.monitor[_note]["loclist"].append(_to)

    def eliminate_location(self, _note, _location):
        self.monitor[_note]["loclist"].remove(_location)
        if not self.monitor[_note]["loclist"]:
            del self.monitor[_note]

    def eliminate_note(self, _note):
        del self.monitor[_note]

    def search_note(self, _note):
        if _note in self.monitor:
            return self.monitor[_note]["token"], self.monitor[_note]["loclist"]
        else:
            return 0, []

    def edit_note(self, _note, _newnote):
        self.monitor[_newnote] = copy.deepcopy(_note)
        del self.monitor[_note]

    # hookfunc of monitor
    # ======================================================================================
    def monitor_hook0(self, _note, _token, _worf: ClusterComponent):
        # add token, a global method
        self.monitor[_note] = {}
        self.monitor[_note]["token"] = _token
        self.monitor[_note]["worf"] = _worf
        self.monitor[_note]["loclist"] = []

    def monitor_hook1(
        self, _clusterid: int, _bulk: DataBulk, _clusterlist: list,
    ):
        # hook1, check before read
        ExtraQueryList = []
        note = _bulk.modelname + "_" + str(_bulk.layerindex) + "_" + _bulk.datatype
        worf = self.monitor[note]["worf"]
        if worf == ClusterComponent.WeightBuffer:
            if (
                _clusterlist[_clusterid].WeightBuffer.Scratchpad.check_scratchpad(_bulk)
                > 0
            ):
                return ExtraQueryList
            else:
                qdma = QueryDMA(_bulk, _clusterid, worf)
                ExtraQueryList.append(qdma)
                self.insert_note(note, _clusterid)
        elif worf == ClusterComponent.FeatureBuffer:
            if (
                _clusterlist[_clusterid].FeatureBuffer.Scratchpad.check_scratchpad(
                    _bulk
                )
                == _bulk.bulksizebyte
            ):
                return ExtraQueryList
            else:
                _, loclist = self.search_note(note)
                if not loclist:
                    for loc in loclist:
                        subbulksize = _clusterlist[
                            loc
                        ].FeatureBuffer.Scratchpad.check_scratchpad(_bulk)
                        if subbulksize > 0:
                            qbus = QueryRingBus(
                                _bulk, subbulksize, loc, _clusterid, worf
                            )
                            ExtraQueryList.append(qbus)
                            self.transfer_note(note, loc, _clusterid)
                if not ExtraQueryList:
                    qdma = QueryDMA(_bulk, _clusterid, worf)
                    ExtraQueryList.append(qdma)
                    self.insert_note(note, _clusterid)
        return ExtraQueryList

    def monitor_hook2(
        self, _clusterid: int, _bulk: DataBulk, _clusterlist: list,
    ):  # hook2, check before write
        ExtraQueryList = []
        note = _bulk.modelname + "_" + str(_bulk.layerindex) + "_" + _bulk.datatype
        # assert _worf == ClusterComponent.FeatureBuffer:
        return ExtraQueryList

    def monitor_hook3(
        self, modelname, layerindex, datatype, _clusterlist: list,
    ):  # hook3, check after one layer finish
        note = modelname + "_" + str(layerindex) + "_" + datatype
        self.monitor[note]["token"] -= 1
        if self.monitor[note]["token"] == 0:
            worf = self.monitor[note]["worf"]
            _, loclist = self.search_note(note)
            for loc in loclist:
                if worf == ClusterComponent.WeightBuffer:
                    _clusterlist[loc].WeightBuffer.release(note)
                elif worf == ClusterComponent.FeatureBuffer:
                    _clusterlist[loc].FeatureBuffer.release(note)
            self.eliminate_note(note)

