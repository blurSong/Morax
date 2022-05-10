# Chip class of Morax
# Morax means the mixing of CMOS and RRAM
# Author: Tao Song.
# Last modification: 0327

from ast import Break
from stat import S_ISFIFO
from urllib.request import install_opener
import numpy as np
import subprocess as SP
import multiprocessing as MP
import copy
import re

import morax.system.interface as IF
import morax.system.query as QR
import morax.system.timefilm as TF
import morax.system.config as CFG
import morax.system.memonitor as MM
import morax.system.schedule as SCH
from morax.model.model import ModelDAG
from algorithm import online as OL

import dma_bus
import cluster


class MoraxChip:
    def __init__(self) -> None:
        self.DMA = dma_bus.DMA()
        self.RingBus = dma_bus.RingBus()
        self.ClusterNum = CFG.MoraxConfig.ClusterNum
        self.ClusterList = []
        for clstid in range(self.ClusterNum):
            clst = cluster.MoraxCluster(clstid)
            self.ClusterList.append(copy.deepcopy(clst))
        self.MoraxTileFilm = TF.TimeFilm()

    def invoke_morax(self, _modelDAG: ModelDAG, _monitor: MM.Memonitor):
        # RRAM:
        # CMOS:
        CandidateLayerList = [-1]
        while True:
            # 0.1 choose one layer to run
            # todo
            thisrun_index = OL.schedule_one_layer(
                CandidateLayerList, _modelDAG, self.ClusterList
            )

            # 0.2 update candidate list
            CandidateLayerList.remove(thisrun_index)
            for candiidx in _modelDAG.toVertexDict[thisrun_index]:
                iscandidate = True
                for tmpfromidx in _modelDAG.fromVertexDict[candiidx]:
                    if (
                        tmpfromidx != thisrun_index
                        and not _modelDAG.LayerQueryClassDict[tmpfromidx].FINISHED_FLAG
                    ):
                        iscandidate = False
                        break
                if iscandidate:
                    CandidateLayerList.append(candiidx)

            # 0.3 get layer query
            Query_thislayer = copy.deepcopy(
                _modelDAG.LayerQueryClassDict[thisrun_index]
            )
            assert isinstance(Query_thislayer, QR.LayerQuery)

            # 0.4 hook 0
            token = len(_modelDAG.toVertexDict[thisrun_index])
            onRRAM = len(Query_thislayer.assignment) != 0
            _monitor.hook0_init(
                token,
                Query_thislayer.batch,
                thisrun_index,
                Query_thislayer.layerclass,
                onRRAM,
            )

            # 1.1 pre assignment

            # 2 check and run subquery
            querynum = Query_thislayer.subquerynum
            for subq_index in range(querynum):
                this_subq = Query_thislayer.SubQueryList[subq_index]
                # hook 1
                if isinstance(this_subq, QR.QueryBuffer):
                    if this_subq.execution == IF.BO.Read:
                        _monitor.hook1_cbr()
            return


def schedule_query(_now_t, _query: QR.LayerQuery, _clusterlist: list):
    # TODO: Update greedy algo to more advanced rules
    onRRAM = len(_query.assignment) != 0
    if not onRRAM:  # onCMOS
        layertype = _query.layerclass.layer_type
        if isinstance(layertype, IF.LLT):
            clstnum = CFG.MoraxConfig.ClusterNum
            tcnum = CFG.MoraxConfig.TCNum
            tclast_t_list = []
            for clst in _clusterlist:
                tclast_t_list.append(clst.get_tc_submit_t())

    last_t_list = []
    for clst in self.ClusterList:
        last_t_list.append(clst.ClusterTimeFilm[-1].submit_t)
    issue_t = min(last_t_list)
    return
