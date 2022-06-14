# from morax.hardware.buffer import DataBulk
from morax.system.config import MoraxConfig
from morax.system.query import QueryRingBus, QueryDMA
import copy
from morax.system.interface import *
from morax.frontend.api import (
    get_layer_scratchdict,
    get_datatype,
    get_weight_scratchdict,
    get_weight_datatype,
)

#  a cheat-sheet to record the locations and token of mm_idx_type
class Memonitor:
    def __init__(self) -> None:
        self.scratchpadnum = MoraxConfig.ClusterNum
        self.monitor = {}

    def add_loc(self, _note: str, _location: int):
        if _location not in self.monitor[_note]["loclist"]:
            self.monitor[_note]["loclist"].append(_location)

    def transfer_note(self, _note, _from, _to):
        assert _from in self.monitor[_note]["loclist"]
        if _to not in self.monitor[_note]["loclist"]:
            self.monitor[_note]["loclist"].append(_to)

    def eliminate_loc(self, _note, _location):
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

    # ================================================================================
    # hooks 0517

    def hook0_init(self, _token, _batch, _index, _layerclass, onRRAM=False):
        """ 
        make layer output note, and make offchip note if needed. 
        invoke when run a new layer.
        """

        # make output note
        outnote = (
            _layerclass.modelname + "_" + str(_index) + "_" + get_datatype(_layerclass)
        )
        self.monitor[outnote] = {}
        self.monitor[outnote]["token"] = _token
        self.monitor[outnote]["worf"] = ClusterComponent.FeatureBuffer
        self.monitor[outnote]["loclist"] = []
        layer_scratchdict = copy.deepcopy(get_layer_scratchdict(_layerclass))
        layer_scratchdict["B"] = _batch
        self.monitor[outnote]["layer_scratchdict"] = layer_scratchdict

        # make weight and offchip data not
        (IIleft, IIright) = _layerclass.input_indecies_tuple
        if (IIleft == 0 or IIright == 0) and not onRRAM:
            panote = (
                _layerclass.modelname
                + "_"
                + str(_index)
                + "_"
                + get_weight_datatype(_layerclass)
            )
            self.monitor[panote] = {}
            self.monitor[panote]["token"] = 1
            self.monitor[panote]["worf"] = ClusterComponent.WeightBuffer
            self.monitor[panote]["loclist"] = []
            self.monitor[panote]["layer_scratchdict"] = copy.deepcopy(
                get_weight_scratchdict(_layerclass)
            )

    # hook1, check before read
    def hook1_cbr(
        self, _clusterid: int, _bulk, _chip_clusterlist: list,
    ):
        if _bulk.bulksizebyte == 0:
            return []
        ExtraQueryList = []
        note = _bulk.modelname + "_" + str(_bulk.layerindex) + "_" + _bulk.datatype
        tok, _ = self.search_note(note)
        if tok == 0:
            worf = ClusterComponent.FeatureBuffer
        else:
            worf = self.monitor[note]["worf"]
        if worf == ClusterComponent.WeightBuffer:
            if (
                _chip_clusterlist[_clusterid].WeightBuffer.Scratchpad.check_scratchpad(
                    _bulk
                )
                > 0
            ):
                return ExtraQueryList
            else:
                qdma = QueryDMA(_bulk, _clusterid, worf)
                ExtraQueryList.append(qdma)
                self.insert_note(note, _clusterid)
        elif worf == ClusterComponent.FeatureBuffer:
            if (
                _chip_clusterlist[_clusterid].FeatureBuffer.Scratchpad.check_scratchpad(
                    _bulk
                )
                == _bulk.bulksizebyte
            ):
                return ExtraQueryList
            else:
                _, loclist = self.search_note(note)
                if len(loclist) > 0:
                    for loc in loclist:
                        subbulksize = _chip_clusterlist[
                            loc
                        ].FeatureBuffer.Scratchpad.check_scratchpad(_bulk)
                        if subbulksize > 0:
                            qbus = QueryRingBus(
                                _bulk, subbulksize, loc, _clusterid, worf
                            )
                            ExtraQueryList.append(qbus)
                            self.transfer_note(note, loc, _clusterid)
                else:  # if not ExtraQueryList:
                    # NOTE do not record this in strachpad and memonitor
                    qdma = QueryDMA(_bulk, _clusterid, worf)
                    ExtraQueryList.append(qdma)
                    # self.insert_note(note, _clusterid)
        return ExtraQueryList

    # hook2, check before write
    def hook2_cbw(
        self, _clusterid: int, _index, _layerclass,
    ):
        outnote = (
            _layerclass.modelname + "_" + str(_index) + "_" + get_datatype(_layerclass)
        )
        self.add_loc(outnote, _clusterid)
        return

    # hook3, check input after one layer finish
    def hook3_caf(
        self, pre_index, _pre_layerclass, _clusterlist: list,
    ):
        note = (
            _pre_layerclass.modelname
            + "_"
            + str(pre_index)
            + "_"
            + get_datatype(_pre_layerclass)
        )
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
