# DMA class of Morax
# Morax means the mixing of CMOS and RRAM
# Author: Tao Song.
# Last modification: 0328

from http.client import CONTINUE
import subprocess as SP
import multiprocessing as MP
import copy
from morax.system.interface import TO, ClusterComponent
from morax.system.timefilm import TimeFilm, TimeStamp
from morax.system.config import MoraxConfig, HWParam
from morax.system.query import QueryBuffer, QueryRingBus, QueryDMA
from morax.hardware.buffer import DataBulk


# read label, transfer bulk
class ClusterTransfeActionDict:
    def __init__(self, _databulk: DataBulk, _fromCluster, _toCluster) -> None:
        self.fromCluster = _fromCluster
        self.toCluster = _toCluster
        self.bulknote = (
            _databulk.modelname
            + "_"
            + str(_databulk.layerindex)
            + "_"
            + _databulk.datatype
        )


class DRAMReadActionDict:
    def __init__(self, _databulk: DataBulk, _toCluster) -> None:
        self.toCluster = _toCluster
        self.bulknote = (
            _databulk.modelname
            + "_"
            + str(_databulk.layerindex)
            + "_"
            + _databulk.datatype
        )


class RingBus:
    def __init__(self) -> None:
        self.TimeFilm = TimeFilm()
        self.ClusterTransferList = []

    def run_query(self, _q_bus: QueryRingBus, _issue_t):
        busatd = ClusterTransfeActionDict(
            _q_bus.databulkclass, _q_bus.fromCluster, _q_bus.toCluster
        )
        busts = TimeStamp(TO.ClusterTransfer, _issue_t, _q_bus.databulkclass.label)
        hoptime = _q_bus.subbulksizebyte * 8 // MoraxConfig.ClusterBusBandwidthGbps
        toc = _q_bus.toCluster
        fromc = _q_bus.fromCluster
        if toc > fromc:
            hops = min(toc - fromc, fromc + MoraxConfig.ClusterNum - toc)
        else:
            hops = min(fromc - toc, toc + MoraxConfig.ClusterNum - fromc)
        runtime = hoptime * hops
        busts.update_span(runtime)
        self.TimeFilm.append_stamp(busts)
        self.ClusterTransferList.append(busatd)
        # return busts.submit_t
        return self.TimeFilm[-1].submit_t

    def run_query_then_write_buffer(self, _q_buffer: QueryBuffer, _clusterlist):
        # attached to BUS queries list
        if _q_buffer.locationEnum == ClusterComponent.FeatureBuffer:
            _clusterlist[_q_buffer.clusterid].FeatureBuffer.write_buffer(
                _q_buffer.databulkclass
            )
        else:
            _clusterlist[_q_buffer.clusterid].WeightBuffer.write_buffer(
                _q_buffer.databulkclass
            )
        return


class DMA:
    def __init__(self) -> None:
        self.TimeFilm = TimeFilm()
        self.DRAMReadList = []

    def run_query(self, _q_dma: QueryDMA, _issue_t, _clusterlist) -> int:
        dmaad = DRAMReadActionDict(_q_dma.databulkclass, _q_dma.toCluster)
        dmats = TimeStamp(TO.ReadDRAM, _issue_t, _q_dma.databulkclass.bulklabel)
        runtime = (
            _q_dma.databulkclass.bulksizebyte * 8 // MoraxConfig.OffChipBandwidthGbps
        )
        # if _q_dma.worf == ClusterComponent.FeatureBuffer:
        # NOTE DO NOTHING!
        # _clusterlist[_q_dma.toCluster].FeatureBuffer.write_buffer(
        #     _q_dma.databulkclass
        # )
        if _q_dma.worf == ClusterComponent.WeightBuffer:
            _clusterlist[_q_dma.toCluster].WeightBuffer.write_buffer(
                _q_dma.databulkclass
            )
        dmats.update_span(runtime)
        self.TimeFilm.append_stamp(dmats)
        self.DRAMReadList.append(dmaad)
        # return dmats.submit_t
        return self.TimeFilm[-1].submit_t
