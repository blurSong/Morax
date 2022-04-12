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
        return True

    def readANote(self, _bulk: DataBulk):
        if check_scratchpad(_bulk):
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

    def monitor_hook0(self, _note, _token, _worf: ClusterComponent):
        # add token, a global method
        self.monitor[_note] = {}
        self.monitor[_note]["token"] = _token
        self.monitor[_note]['worf'] = _worf
        self.monitor[_note]['loclist'] = []

    def insert_note(self, _note: str, _location: int):
        '''
        if _note not in self.monitor:
            self.monitor[_note]["loclist"] = [_location]
            # self.monitor[_note]["token"] = _token
            assert _location in range(self.scratchpadnum)  # -1 for offchip (depr)
        else:
            # assert _token == self.monitor[_note]["token"]
        '''
        self.monitor[_note]["loclist"].append(_location)

    def transfer_note(self, _note, _from, _to):
        assert _from in self.monitor[_note]["loclist"]
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

    def monitor_hook1(
        self,
        _clusterid: int,
        _bulk: DataBulk,
        _clusterlist: list,
    ):
        # hook1, check before read
        ExtraQueryList = []
        note = _bulk.modelname + "_" + str(_bulk.layerindex) + "_" + _bulk.datatype
        worf = self.monitor[note]["worf"]
        if worf == ClusterComponent.WeightBuffer:
            if _clusterlist[_clusterid].WeightBuffer.Scratchpad.check_scratchpad(_bulk):
                return ExtraQueryList
            else:
                qdma = QueryDMA(_bulk, _clusterid)
                ExtraQueryList.append(qdma)
                self.insert_note(note, 1, _clusterid)
        elif worf == ClusterComponent.FeatureBuffer:
            if _clusterlist[_clusterid].FeatureBuffer.Scratchpad.check_scratchpad(_bulk):
                return ExtraQueryList
            else:
                _, loclist = self.search_note(note)
                for loc in loclist:
                    if _clusterlist[loc].FeatureBuffer.Scratchpad.check_scratchpad(_bulk):
                        qbus = QueryRingBus(_bulk, loc, _clusterid)
                        ExtraQueryList.append(qbus)
                        self.transfer_note(note, loc, _clusterid)
                        # todo update monitor
                if not ExtraQueryList:
                    qdma = QueryDMA(_bulk, _clusterid)
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
    )   # hook3, check after one layer finish 
        note = modelname + "_" + str(layerindex) + "_" + datatype
        self.monitor[note]['token'] -= 1
        if self.monitor[note]['token'] == 0:
            worf = self.monitor[note]["worf"]
            _, loclist = self.search_note(note)
            for loc in loclist:
                if worf == ClusterComponent.WeightBuffer:
                    _clusterlist[loc].WeightBuffer.release(note)
                elif worf == ClusterComponent.FeatureBuffer:
                    _clusterlist[loc].FeatureBuffer.release(note)
            self.eliminate_note(note)







