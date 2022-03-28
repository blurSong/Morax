# Cluster class of Morax
# Morax means the mixing of CMOS and RRAM
# Author: Tao Song.
# Last modification: 0327

import numpy as np
import subprocess as SP
import multiprocessing as MP
import copy
import re
from morax.system.interface import TO, ClusterComponent
from morax.system.timefilm import TimeFilm, TimeStamp
from morax.system.config import MoraxConfig, HWParam
from morax.system.query import QueryRingBus, QueryDMA
from buffer import DataBulk


# read label, transfer bulk
class ClusterTransfeActionDict:
    def __init__(self, _databulk: DataBulk, _fromCluster, _toCluster) -> None:
        self.fromCluster = _fromCluster
        self.toCluster = _toCluster
        self.databulk = copy.deepcopy(_databulk)


class DRAMReadActionDict:
    def __init__(self, _databulk: DataBulk, _toCluster) -> None:
        self.toCluster = _toCluster
        self.databulk = copy.deepcopy(_databulk)


class RingBus:
    def __init__(self) -> None:
        self.TimeFilm = TimeFilm()
        self.ClusterTransferList = []

    def run_query(self, _q_clutrans: QueryRingBus, _issue_t):
        busatd = ClusterTransfeActionDict(
            _q_clutrans.databulkclass, _q_clutrans.fromCluster, _q_clutrans.toCluster
        )
        busts = TimeStamp(TO.ClusterTransfer, _issue_t, _q_clutrans.databulkclass.label)
        runtime = (
            _q_clutrans.databulkclass.sizebyte
            * MoraxConfig.PrecisionBits
            / MoraxConfig.ClusterBusBandwidthGbps
        )
        busts.update_span(runtime)
        self.TimeFilm.append_stamp(busts)
        self.ClusterTransferList.append(busatd)
        return busts.submit_t


class DMA:
    def __init__(self) -> None:
        self.TimeFilm = TimeFilm()
        self.DRAMReadList = []

    def run_query(self, _q_dma: QueryDMA, _issue_t) -> int:
        dmaad = DRAMReadActionDict(_q_dma.databulkclass, _q_dma.toCluster)
        dmats = TimeStamp(TO.ReadDRAM, _issue_t, _q_dma.databulkclass.label)
        runtime = (
            _q_dma.databulkclass.sizebyte
            * MoraxConfig.PrecisionBits
            / MoraxConfig.OffChipBandwidthGbps
        )
        dmats.update_span(runtime)
        self.TimeFilm.append_stamp(dmats)
        self.DRAMReadList.append(dmaad)
        return dmats.submit_t
