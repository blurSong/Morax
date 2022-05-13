# Chip class of Morax
# Morax means the mixing of CMOS and RRAM
# Author: Tao Song.
# Last modification: 0327

from ast import Break
from http.client import CONTINUE
from stat import S_ISFIFO
from urllib.request import install_opener
import numpy as np
import subprocess as SP
import multiprocessing as MP
import copy
import re

from pyrsistent import T

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

            # 1 check and run subquery
            used_cluster_id = []
            extra_queries = []
            querynum = Query_thislayer.subquerynum
            while Query_thislayer:
                querybulk = spcify_querybulk(Query_thislayer)
                this_clusterid = schedule_a_cluster(querybulk, self.ClusterList)

            for q_index in range(querynum):
                this_query = Query_thislayer.SubQueryList[q_index]
                # hook 1
                if isinstance(this_query, QR.QueryBuffer):
                    if this_query.execution == IF.BO.Read:
                        this_clusterid = schedule_a_cluster(
                            Query_thislayer, self.ClusterList
                        )
                        used_cluster_id.append(this_clusterid)
                        this_query.update_clusterid(this_clusterid)
                        extra_queries = _monitor.hook1_cbr(
                            this_query.clusterid,
                            this_query.databulkclass,
                            self.ClusterList,
                        )
                    elif this_query.execution == IF.BO.Write:
                        this_query.update_clusterid(this_clusterid)

            return


def schedule_a_cluster(
    _querylist: list, _clusterlist: list, _layerqueryclass: QR.LayerQuery
) -> int:
    # TODO: Update greedy algo to more advanced rule
    for q in _querylist:
        if isinstance(q, QR.QueryBuffer):
            continue
        else:
            if isinstance(q, QR.QueryExcuteOnNVTC):
                PLACE = IF.CC.nvTensorCore
            elif isinstance(q, QR.QueryExcuteOnVPU):
                PLACE = IF.CC.VPU
            elif isinstance(q, QR.QueryExcuteOnTC):
                PLACE = IF.CC.TensorCore
            elif isinstance(q, QR.QueryExcuteOnSMU):
                PLACE = IF.CC.SMU
            break
    if PLACE == IF.CC.TensorCore:
        tclast_t_list = []
        for clst in _clusterlist:
            tclast_t_list.append(min(clst.report_tc_submit_t()))
            his_clstid = tclast_t_list.index(min(tclast_t_list))
    elif PLACE == IF.CC.VPU:
        vpulast_t_list = []
        for clst in _clusterlist:
            vpulast_t_list.append(clst.report_vpu_submit_t())
        this_clstid = vpulast_t_list.index(min(vpulast_t_list))
    elif PLACE == IF.CC.nvTensorCore:
        # dict of {clstid: [tuple1(nvtcid, sliceidlist), tuple2, ... ]}
        clstid_list = []
        assignment_doclotnsl = _layerqueryclass.assignment
        for clstid, _ in assignment_doclotnsl.items():
            clstid_list.append(clstid)
        this_clstid = min(clstid_list)  # why
    return this_clstid


def spcify_querybulk(_query_list: list):
    assert isinstance(_query_list[0], QR.QueryBuffer)
    assert _query_list[0].execution == IF.BO.Read
    this_querybulk = []
    EXE_FLAG = 0
    while True:
        this_querybulk.append(copy.deepcopy(_query_list.pop(index=0)))
        if EXE_FLAG == 0:
            if isinstance(_query_list[0], QR.QueryExcute):
                EXE_FLAG = 1
        else:
            if len(_query_list) == 0:
                break
            elif isinstance(_query_list[0], QR.VritualQuerySeparator):
                _query_list.pop(index=0)
                break
            elif isinstance(_query_list[0], QR.QueryBuffer):
                if _query_list[0].execution == IF.BO.Read:
                    break
    return this_querybulk
