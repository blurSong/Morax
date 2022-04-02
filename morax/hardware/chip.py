# Chip class of Morax
# Morax means the mixing of CMOS and RRAM
# Author: Tao Song.
# Last modification: 0327

import numpy as np
import subprocess as SP
import multiprocessing as MP
import copy
import re
from morax.system.timefilm import TimeFilm, TimeStamp
from morax.system.config import MoraxConfig, HWParam
import morax.system.query as Q
import buffer
import nvtensorcore
import tensorcore
import smu
import vpu
import dma_bus
import cluster


class MoraxChip:
    def __init__(self) -> None:
        self.DMA = dma_bus.DMA()
        self.RingBus = dma_bus.RingBus()
        self.ClusterNum = MoraxConfig.ClusterNum
        self.ClusterList = []
        for clstid in range(self.ClusterNum):
            clst = cluster.MoraxCluster(clstid)
            self.ClusterList.append(copy.deepcopy(clst))
        self.MoraxTileFilm = TimeFilm()

    def invoke_morax(self, _q_layer: Q.LayerQuery, _issue_t: int):
        # RRAM:
        # CMOS: 最小的调度单位为batch， 同一个TC运行到完成分块的所有batch
