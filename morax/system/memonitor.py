from morax.hardware.buffer import DataBulk
from morax.model.layer import LinearLayerType, NonlinearLayerType
from morax.system.config import MoraxConfig, HWParam
from morax.system.query import QueryRingBus, QueryDMA
import copy
from morax.system.interface import *
import numpy as np
from morax.frontend.api import (
    get_layer_scratchdict,
    get_datatype,
    get_weight_scratchdict,
    get_weight_datatype,
)


def edit_data_index(_thisidx, _thatidx, datatype):
    # for concat or append
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


def isrange(c0, c1):
    return c0 >= 0 and c1 >= 0


""" a scratchpad note is:
|-------------------------------------|
| note                                |
| [modelname + layerindex + datatype] |
|-------------------------------------|
| sizebyte                            |          
| datatype                            |
| total_scratchdict                   | # added 0417 merge.api
|-------------------------------------|
| bulklabel1 | sizebyte | scratchdict |
| bulklabel2 | sizebyte | scratchdict |
| bulklabel3 | sizebyte | scratchdict |
|-------------------------------------|
"""


class BulkNote:
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
            bulknote = BulkNote(_bulk)
            if bulknote not in self.Scratchpad[note]["bulknotelist"]:
                self.Scratchpad[note]["bulknotelist"].append(copy.deepcopy(bulknote))
            # assert self.Scratchpad[note]["token"] == _bulk.token
            # NOTE token is maintained in monitor
        else:
            pad = {}
            pad["sizebyte"] = _bulk.bulksizebyte
            pad["datatype"] = _bulk.datatype
            abulknote = bulknote(_bulk)
            pad["bulknotelist"] = [abulknote]
            self.Scratchpad[note] = copy.deepcopy(pad)

    def merge_2scratch(_hw1, _hw2):
        htup1 = _hw1[0]
        htup2 = _hw2[0]
        wtup1 = _hw1[1]
        wtup2 = _hw2[1]
        hw = []
        if abs(htup1[1] - htup2[0]) < MoraxConfig.PEArraySize:
            hw[0] = (htup1[0], htup2[1])
        elif abs(htup1[0] - htup2[1]) < MoraxConfig.PEArraySize:
            hw[0] = (htup2[0], htup1[1])
        else:
            print("merge scratch fatal")
        if abs(wtup1[1] - wtup2[0]) < MoraxConfig.PEArraySize:
            hw[0] = (wtup1[0], wtup2[1])
        elif abs(wtup1[0] - wtup2[1]) < MoraxConfig.PEArraySize:
            hw[0] = (wtup2[0], wtup1[1])
        else:
            print("merge scratch fatal")
        return tuple(hw)

    def merge_scratchpad(self, _note, _layer_scratchdict):
        datatype = self.Scratchpad[_note]["datatype"]
        scratchpad_dict = {}
        layersize_dict = {}
        clist_os, hwdict_os = [], {}  # ((h, h), (w, w))
        hwdict_rram = {}
        blist_vec, vdict_vec = [], {}
        blist_mat, mndict_mat = [], {}
        if datatype == "WET":
            return
            # KCRS, Usually No need to merge
        elif datatype == "FTR":
            # BCHW: OS HWtupBintCinttup  RRAM or Sys BHWintCtup
            for bulknote in self.Scratchpad[_note]["bulknotelist"]:
                b, c, h, w = (
                    bulknote.scratchdict["B"],
                    bulknote.scratchdict["C"],
                    bulknote.scratchdict["H"],
                    bulknote.scratchdict["W"],
                )
                if (
                    isinstance(b, int)
                    and isinstance(c, (int, tuple))
                    and isinstance(h, tuple)
                    and isinstance(w, tuple)
                ):  # _fullsize
                    fullsize = _layer_scratchdict["H"]
                    if isinstance(c, tuple):
                        for cidx in range(c[0], c[1] + 1):
                            if cidx in clist_os:
                                assert cidx == c[0]
                                hwdict_os[cidx] = self.merge_2scratch(
                                    hwdict_os[cidx], (h, w)
                                )
                            else:
                                assert cidx != 0
                                if cidx != c[1]:
                                    clist_os.append(cidx)
                                    hwdict_os[cidx] = ((0, fullsize), (0, fullsize))
                                else:
                                    clist_os.append(cidx)
                                    hwdict_os[cidx] = ((0, h[1]), (0, w[1]))
                    else:
                        hwdict_os[c] = self.merge_2scratch(hwdict_os[c], (h, w))
                elif (
                    isinstance(b, int)
                    and isinstance(c, tuple)
                    and isinstance(h, int)
                    and isinstance(w, int)
                ):  # _chsize
                    chsize = _layer_scratchdict["C"]
                    # if len(c) == _chsize:
                    if c not in hwdict_rram:
                        hwdict_rram[c] = ((h, h), (w, w))
                    else:
                        hwdict_rram[c] = self.merge_2scratch(hwdict_rram[c], (h, w))
        elif datatype == "VEC":
            for bulknote in self.Scratchpad[_note]["bulknotelist"]:
                # B M
                m = bulknote.scratchdict["M"]  # tup
                b = bulknote.scratchdict["B"]  # int
                if b not in blist_vec:
                    blist_vec.append(b)
                    vdict_vec[b] = m
                else:
                    if vdict_vec[b][0] > m[1]:
                        vdict_vec[b] = (m[0], vdict_vec[b][1])
                    elif vdict_vec[b][1] < m[0]:
                        vdict_vec[b] = (vdict_vec[b][0], m[1])
        elif datatype == "MAT":
            for bulknote in self.Scratchpad[_note]["bulknotelist"]:
                # B M N
                m = bulknote.scratchdict["M"]  # tup
                b = bulknote.scratchdict["B"]  # int
                n = bulknote.scratchdict["N"]  # tup
                if b not in blist_mat:
                    blist_mat.append(b)
                    mndict_mat[b] = (m, n)
                else:
                    mndict_mat[b] = self.merge_2scratch(mndict_mat[b], (m, n))
        # End for
        # make note dict
        bsize = _layer_scratchdict["B"]
        if datatype == "FTR":
            if len(hwdict_os) != 0:
                # clist_os, hwdict_os = [], {}
                clist_os.sort()
                scratchpad_dict["B"] = (0, bsize - 1)
                scratchpad_dict["C"] = (clist_os[0], clist_os[-1])
                scratchpad_dict["H"] = (
                    hwdict_os[clist_os[0]][0][0],
                    hwdict_os[clist_os[-1]][0][1],
                )
                scratchpad_dict["W"] = (
                    hwdict_os[clist_os[0]][1][0],
                    hwdict_os[clist_os[-1]][1][1],
                )
            elif len(hwdict_rram) != 0:
                scratchpad_dict["B"] = (0, bsize - 1)
                chs = list(hwdict_rram.keys())
                if len(chs) == 1:
                    scratchpad_dict["C"] = chs[0]
                else:
                    fsize = _layer_scratchdict["H"]
                    chs.sort()
                    scratchpad_dict["C"] = (chs[0][0], chs[-1][1])
                    scratchpad_dict["H"] = (0, fsize - 1)  # hwdict_rram[chs[0]][0]
                    scratchpad_dict["W"] = (0, fsize - 1)  # hwdict_rram[chs[0]][1]
        elif datatype == "VEC":
            vsize = _layer_scratchdict["M"]
            blist_vec.sort()
            scratchpad_dict["B"] = (blist_vec[0], blist_vec[-1])
            scratchpad_dict["M"] = (0, vsize - 1)  # vdict_vec[blist_vec[0]]
        elif datatype == "MAT":
            msize = _layer_scratchdict["M"]
            nsize = _layer_scratchdict["N"]
            blist_mat.sort()
            scratchpad_dict["B"] = (blist_mat[0], blist_mat[-1])
            # scratchpad_dict['M'] = (0, msize-1)  # vdict_vec[blist_vec[0]]
            # scratchpad_dict['N'] = (0, nsize-1)  # vdict_vec[blist_vec[0]]
            scratchpad_dict["M"] = (
                mndict_mat[blist_mat[0]][0][0],
                mndict_mat[blist_mat[-1]][0][1],
            )
            scratchpad_dict["N"] = (
                mndict_mat[blist_mat[0]][1][0],
                mndict_mat[blist_mat[-1]][1][1],
            )
        self.Scratchpad[_note]["scratchpad_dict"] = copy.deepcopy(scratchpad_dict)
        # self.Scratchpad[_note]['layersize_dict'] = copy.deepcopy(_layer_scratchdict)
        return

    def check_scratchpad_deprecated(self, _bulk: DataBulk):
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
            elif _bulk.datatype == "FTR":
                blist = []
                clist = []
                hlist = []
                wlist = []
                for bulknote in self.Scratchpad[note]["bulknotelist"]:
                    blist.append(bulknote.scratchdict["B"])
                    clist.append(bulknote.scratchdict["C"])
                    hlist.append(bulknote.scratchdict["H"])
                    wlist.append(bulknote.scratchdict["W"])
                bchwlist = list(
                    zip(blist, clist, hlist, wlist)
                )  # [((b0,b1), (c0,c1), (h0,h1), (w0,w1)), ..., ]
                bchwlist.sort(key=takeone, reverse=False)
                subbulksize = 0
                for bchwtup in bchwlist:
                    b00, b11 = check_range(
                        _bulk.bulkscratch["B"][0], _bulk.bulkscratch["B"][1], bchwtup[0]
                    )
                    c00, c11 = check_range(
                        _bulk.bulkscratch["C"][0], _bulk.bulkscratch["C"][1], bchwtup[1]
                    )
                    h00, h11 = check_range(
                        _bulk.bulkscratch["H"][0], _bulk.bulkscratch["H"][1], bchwtup[2]
                    )
                    w00, w11 = check_range(
                        _bulk.bulkscratch["W"][0], _bulk.bulkscratch["W"][1], bchwtup[3]
                    )
                    if (
                        isrange(b00, b11)
                        and isrange(c00, c11)
                        and isrange(h00, h11)
                        and isrange(w00, w11)
                    ):
                        subbulksize += _bulk.bulksizebyte * (
                            (c11 - c00 + 1)
                            * (b00 - b11 + 1)
                            * (h00 - h11 + 1)
                            * (w00 - w11 + 1)
                            * 1.0
                            / (
                                (
                                    _bulk.bulkscratch["B"][1]
                                    - _bulk.bulkscratch["B"][0]
                                    + 1
                                )
                                * (
                                    _bulk.bulkscratch["C"][1]
                                    - _bulk.bulkscratch["C"][0]
                                    + 1
                                )
                                * (
                                    _bulk.bulkscratch["H"][1]
                                    - _bulk.bulkscratch["H"][0]
                                    + 1
                                )
                                * (
                                    _bulk.bulkscratch["W"][1]
                                    - _bulk.bulkscratch["W"][0]
                                    + 1
                                )
                            )
                        )
            elif _bulk.datatype == "VEC":
                blist = []
                mlist = []
                for bulknote in self.Scratchpad[note]["bulknotelist"]:
                    blist.append(bulknote.scratchdict["B"])
                    mlist.append(bulknote.scratchdict["M"])
                bmlist = list(zip(blist, mlist))
                bmlist.sort(key=takeone, reverse=False)
                subbulksize = 0
                for bmtup in bmlist:
                    b00, b11 = check_range(
                        _bulk.bulkscratch["B"][0], _bulk.bulkscratch["B"][1], bmtup[0]
                    )
                    m00, m11 = check_range(
                        _bulk.bulkscratch["M"][0], _bulk.bulkscratch["M"][1], bmtup[1]
                    )
                    if isrange(b00, b11) and isrange(m00, m11):
                        subbulksize += _bulk.bulksizebyte * (
                            (m11 - m00 + 1)
                            * (b00 - b11 + 1)
                            * 1.0
                            / (
                                (
                                    _bulk.bulkscratch["B"][1]
                                    - _bulk.bulkscratch["B"][0]
                                    + 1
                                )
                                * (
                                    _bulk.bulkscratch["M"][1]
                                    - _bulk.bulkscratch["M"][0]
                                    + 1
                                )
                            )
                        )
            elif _bulk.datatype == "MAT":
                blist = []
                mlist = []
                nlist = []
                for bulknote in self.Scratchpad[note]["bulknotelist"]:
                    blist.append(bulknote.scratchdict["B"])
                    mlist.append(bulknote.scratchdict["M"])
                    nlist.append(bulknote.scratchdict["N"])
                bmnlist = list(zip(blist, mlist))
                bmnlist.sort(key=takeone, reverse=False)
                subbulksize = 0
                for bmntup in bmnlist:
                    b00, b11 = check_range(
                        _bulk.bulkscratch["B"][0], _bulk.bulkscratch["B"][1], bmntup[0]
                    )
                    m00, m11 = check_range(
                        _bulk.bulkscratch["M"][0], _bulk.bulkscratch["M"][1], bmntup[1]
                    )
                    n00, n11 = check_range(
                        _bulk.bulkscratch["N"][0], _bulk.bulkscratch["N"][1], bmntup[2]
                    )
                    if isrange(b00, b11) and isrange(m00, m11) and isrange(n00, n11):
                        subbulksize += _bulk.bulksizebyte * (
                            (m11 - m00 + 1)
                            * (b00 - b11 + 1)
                            * (n00 - n11 + 1)
                            * 1.0
                            / (
                                (
                                    _bulk.bulkscratch["B"][1]
                                    - _bulk.bulkscratch["B"][0]
                                    + 1
                                )
                                * (
                                    _bulk.bulkscratch["M"][1]
                                    - _bulk.bulkscratch["M"][0]
                                    + 1
                                )
                                * (
                                    _bulk.bulkscratch["N"][1]
                                    - _bulk.bulkscratch["N"][0]
                                    + 1
                                )
                            )
                        )
            return subbulksize

    def check_scratchpad(self, _bulk: DataBulk):
        # return subbulk size only
        note = _bulk.modelname + "_" + str(_bulk.layerindex) + "_" + _bulk.datatype
        if note not in self.Scratchpad:
            return 0
        else:
            scratchpad_dict = self.Scratchpad[note]["scratchpad_dict"]
            transmission_bulksize = 0
            if _bulk.datatype == "WET":
                ktup = _bulk.bulkscratch["K"]
                ctup = _bulk.bulkscratch["C"]
                if isinstance(ktup, tuple):
                    klist = range(ktup[0], ktup[1] + 1)
                else:
                    klist = [ktup]
                lenk = len(klist)
                for kidx in klist:
                    if (
                        scratchpad_dict["K"][0] <= kidx
                        and scratchpad_dict["K"][1] >= kidx
                    ):
                        # if scratchpad_dict['C'][0] <= ctup[0] and scratchpad_dict['C'][1] >= ctup[1]:
                        if isrange(check_range(ctup[0], ctup[1], scratchpad_dict["C"])):
                            transmission_bulksize += _bulk.bulksizebyte / lenk
            elif _bulk.datatype == "FTR":
                # BCHW
                bint = _bulk.bulkscratch["B"]
                ctup = _bulk.bulkscratch["C"]
                htup = _bulk.bulkscratch["H"]
                wtup = _bulk.bulkscratch["W"]
                # assert isinstance(bint, int)
                if isinstance(ctup, tuple):
                    crange = check_range(ctup[0], ctup[1], scratchpad_dict["C"])
                else:
                    crange = check_range(ctup, ctup, scratchpad_dict["C"])
                if scratchpad_dict["B"][0] <= bint and scratchpad_dict["B"][1] >= bint:
                    if isrange(crange):
                        ch0 = crange[0]
                        ch1 = crange[1]
                        entcnum = crange[1] - crange[0] + 1
                        if ch0 == scratchpad_dict["C"][0]:
                            entcnum -= 1
                            hb0 = (
                                htup[0]
                                if htup[0] > scratchpad_dict["H"][0]
                                else scratchpad_dict["H"][0]
                            )
                            transmission_bulksize += (
                                _bulk.bulksizebyte / (ctup[1] - ctup[0] + 1)
                            ) * ((htup[1] - hb0 + 1) / (htup[1] - htup[0] + 1))
                        if ch1 == scratchpad_dict["C"][1]:
                            entcnum -= 1
                            hb1 = (
                                htup[1]
                                if htup[1] < scratchpad_dict["H"][1]
                                else scratchpad_dict["H"][1]
                            )
                            transmission_bulksize += (
                                _bulk.bulksizebyte / (ctup[1] - ctup[0] + 1)
                            ) * ((hb1 - htup[0] + 1) / (htup[1] - htup[0] + 1))
                        if entcnum > 0:
                            transmission_bulksize += (
                                _bulk.bulksizebyte / (ctup[1] - ctup[0] + 1)
                            ) * entcnum
            elif _bulk.datatype == "VEC":
                # BM
                bint = _bulk.bulkscratch["B"]
                mtup = _bulk.bulkscratch["M"]
                if scratchpad_dict["B"][0] <= bint and scratchpad_dict["B"][1] >= bint:
                    transmission_bulksize += _bulk.bulksizebyte
            elif _bulk.datatype == "MAT":
                # BMN
                bint = _bulk.bulkscratch["B"]
                mtup = _bulk.bulkscratch["M"]
                ntup = _bulk.bulkscratch["N"]
                if scratchpad_dict["B"][0] <= bint and scratchpad_dict["B"][1] >= bint:
                    if bint == scratchpad_dict["B"][0]:
                        m0 = (
                            mtup[0]
                            if mtup[0] > scratchpad_dict["M"][0]
                            else scratchpad_dict["M"][0]
                        )
                        m1 = mtup[1]
                    elif bint == scratchpad_dict["B"][1]:
                        m0 = mtup[0]
                        m1 = (
                            mtup[1]
                            if mtup[1] < scratchpad_dict["M"][1]
                            else scratchpad_dict["M"][1]
                        )
                    else:
                        m0 = mtup[0]
                        m1 = mtup[1]
                    transmission_bulksize += (
                        _bulk.bulksizebyte * (m1 - m0 + 1) / (mtup[1] - mtup[0] + 1)
                    )
            # endif
            return transmission_bulksize

    def readANote(self, _bulk: DataBulk):
        if self.check_scratchpad(_bulk) > 0:
            return "Success"
        else:
            return "Fail"

    def get_size(self, _note):
        if _note in self.Scratchpad:
            return self.Scratchpad[_note]["sizebyte"]
        else:
            return 0

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
        self, _clusterid: int, _bulk: DataBulk, _chip_clusterlist: list,
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
