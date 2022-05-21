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
from regex import R

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
        self.MoraxTimeFilm = TF.TimeFilm()

    def invoke_morax(self, _modelDAG: ModelDAG, _monitor: MM.Memonitor):
        # RRAM:
        # CMOS:
        CandidateLayerList = [-1]
        while True:
            # 0.1 choose one layer, report layer index and issue_time
            # todo
            thisrun_index, LAYER_ISSUE_TIME = OL.schedule_one_layer(
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
            ThisLayerQuery = _modelDAG.LayerQueryClassDict[thisrun_index]
            assert isinstance(ThisLayerQuery, QR.LayerQuery)
            ThisLayerQuery.set_issue_t(LAYER_ISSUE_TIME)

            # 0.4 hook 0
            token = len(_modelDAG.toVertexDict[thisrun_index])
            onRRAM = len(ThisLayerQuery.assignment) != 0
            _monitor.hook0_init(
                token,
                ThisLayerQuery.batch,
                thisrun_index,
                ThisLayerQuery.layerclass,
                onRRAM,
            )

            # ready2go
            used_cluster_id = []
            extra_queries = []
            subquerylist = copy.deepcopy(ThisLayerQuery.SubQueryList)
            ISSUE_T = LAYER_ISSUE_TIME
            SUBMIT_T = ISSUE_T
            SUBMIT_T_0 = ISSUE_T
            while subquerylist:
                """"""
                # spcify query bulks and clst id
                subquerybulk = spcify_querybulk(subquerylist)
                this_clusterid = schedule_a_cluster(
                    subquerybulk, self.ClusterList, ThisLayerQuery
                )
                used_cluster_id.append(this_clusterid)
                """"""
                # run this q bulk
                for q_sub_idx in range(len(subquerybulk)):
                    this_subquery = subquerybulk[q_sub_idx]
                    extra_queries.clear()
                    # 1.1 hook 1 2
                    if isinstance(this_subquery, QR.QueryBuffer):
                        if this_subquery.execution == IF.BO.Read:
                            if not this_subquery.location_assigned():
                                this_subquery.update_clusterid(this_clusterid)
                            extra_queries = _monitor.hook1_cbr(
                                this_subquery.clusterid,
                                this_subquery.databulkclass,
                                self.ClusterList,
                            )
                        elif this_subquery.execution == IF.BO.Write:
                            this_subquery.update_clusterid(this_clusterid)
                            _monitor.hook2_cbw(
                                this_subquery.clusterid,
                                thisrun_index,
                                ThisLayerQuery.layerclass,
                            )
                    """"""
                    # 1.2 run extra_queries
                    EXTRA_T = LAYER_ISSUE_TIME
                    if extra_queries:
                        bus_invoke = False
                        for q_ex in extra_queries:
                            if isinstance(q_ex, QR.QueryDMA):
                                EXTRA_T = self.DMA.run_query(
                                    q_ex, 0, this_subquery.clusterid
                                )
                            elif isinstance(q_ex, QR.QueryRingBus):
                                bus_invoke = True
                                EXTRA_T = self.RingBus.run_query(q_ex, LAYER_ISSUE_TIME)
                        if bus_invoke:
                            self.RingBus.run_query_then_write_buffer(
                                this_subquery, self.ClusterList
                            )
                    """"""
                    # 1.3 run this_subquery
                    if isinstance(this_subquery, QR.QueryBuffer):
                        if this_subquery.execution == IF.BO.Read:
                            ISSUE_T = EXTRA_T
                            SUBMIT_T = self.ClusterList[
                                this_subquery.clusterid
                            ].run_query(this_subquery, ISSUE_T)
                        if this_subquery.execution == IF.BO.Write:
                            ISSUE_T = SUBMIT_T
                            SUBMIT_T = self.ClusterList[
                                this_subquery.clusterid
                            ].run_query(this_subquery, ISSUE_T)
                    elif isinstance(this_subquery, QR.QueryExcute):
                        if isinstance(this_subquery, QR.QueryExcuteOnTC):
                            TC_ISSUE_T = max(
                                self.ClusterList[this_clusterid].report_fb_submit_t(),
                                self.ClusterList[this_clusterid].report_wb_submit_t(),
                            )
                            SUBMIT_T = self.ClusterList[this_clusterid].run_query(
                                this_subquery, TC_ISSUE_T
                            )
                        if isinstance(this_subquery, QR.QueryExcuteOnVPU):
                            ISSUE_T = SUBMIT_T
                            SUBMIT_T = self.ClusterList[this_clusterid].run_query(
                                this_subquery, ISSUE_T
                            )
                        if isinstance(this_subquery, QR.QueryExcuteOnSMU):
                            SUBMIT_T = self.ClusterList[this_clusterid].run_query(
                                this_subquery, ISSUE_T
                            )
                        if isinstance(this_subquery, QR.QueryExcuteOnNVTC):
                            if re.search(this_subquery.dfmod, "LUT"):
                                SUBMIT_T = self.ClusterList[
                                    this_subquery.clusterid
                                ].run_query(this_subquery, ISSUE_T)
                            if re.search(this_subquery.dfmod, "Xbar"):
                                ISSUE_T = SUBMIT_T
                                SUBMIT_T_0 = max(
                                    self.ClusterList[this_subquery.clusterid].run_query(
                                        this_subquery, ISSUE_T
                                    ),
                                    SUBMIT_T_0,
                                )
                                SUBMIT_T = (
                                    SUBMIT_T_0
                                    if isinstance(
                                        subquerybulk[q_sub_idx + 1], QR.QueryExcuteOnVPU
                                    )
                                    else SUBMIT_T
                                )

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
            this_clstid = tclast_t_list.index(min(tclast_t_list))
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
    """
    A q bulk usually is: R, R E W, R E W,...
    A nvtc bulk is R (R), EEEEEEE, V W
    TODO 0520: Use VritualQuerySeparator to spcify bulk
    """
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
