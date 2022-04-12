# Chip class of Morax
# Morax means the mixing of CMOS and RRAM
# Author: Tao Song.
# Last modification: 0327

import numpy as np
import subprocess as SP
import multiprocessing as MP
import copy
import re

import morax.system.interface as IF
import morax.system.query as QR
import morax.system.timefilm as TF
import morax.system.config as CF
import morax.system.memonitor as MM
from morax.model.model import ModelDAG

import dma_bus
import cluster


class MoraxChip:
    def __init__(self) -> None:
        self.DMA = dma_bus.DMA()
        self.RingBus = dma_bus.RingBus()
        self.ClusterNum = CF.MoraxConfig.ClusterNum
        self.ClusterList = []
        for clstid in range(self.ClusterNum):
            clst = cluster.MoraxCluster(clstid)
            self.ClusterList.append(copy.deepcopy(clst))
        self.MoraxTileFilm = TF.TimeFilm()

    def invoke_morax(self, _modeldag: ModelDAG, _monitor: MM.Memonitor):
        # RRAM:
        # CMOS:
        thisrun_index = -1
        CandidateLayerList = []
        while True:
            
