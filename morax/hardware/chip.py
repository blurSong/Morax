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
import morax.system.timefilm as TF
import morax.system.config as CFG
import morax.system.memonitor as MM
import morax.system.query as QR
from morax.model.model import ModelDAG
from algorithm import online as OL

import morax.hardware.dma_bus as DMA_BUS
import morax.hardware.cluster as CLST


class MoraxChip:
    def __init__(self) -> None:
        self.DMA = DMA_BUS.DMA()
        self.RingBus = DMA_BUS.RingBus()
        self.ClusterNum = CFG.MoraxConfig.ClusterNum
        self.ClusterList = []
        for clstid in range(self.ClusterNum):
            clst = CLST.MoraxCluster(clstid)
            self.ClusterList.append(copy.deepcopy(clst))
        self.MoraxTimeFilm = TF.TimeFilm()
        print("[Morax][Hardware] Chip initialized.")

    def invoke_morax(
        self, _modelDAG: ModelDAG, _monitor: MM.Memonitor, Schduler: OL.Schduler
    ):
        # RRAM:
        # CMOS:
        _modelDAG.update_submit_t(-1, 0)
        CandidateLayerList = copy.deepcopy(_modelDAG.toVertexDict[-1])
        while True:
            # 0.1 choose one layer, report layer index and issue_time
            # TODO OL.schedule_one_layer
            thisrun_index, LAYER_ISSUE_TIME = Schduler.schedule_one_layer(
                CandidateLayerList, _modelDAG,
            )

            # 0.2 update candidate list
            CandidateLayerList.remove(thisrun_index)
            _modelDAG.LayerQueryClassDict[thisrun_index].FINISHED_FLAG = True
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
                _modelDAG.modelname,
                onRRAM,
            )

            # ready2go
            used_cluster_id = []
            extra_queries = []
            subquerylist = copy.deepcopy(ThisLayerQuery.SubQueryList)

            while subquerylist:
                """"""
                # spcify query bulks and clst id
                subquerybulk = spcify_querybulk(subquerylist)
                this_clusterid = schedule_a_cluster(
                    subquerybulk, self.ClusterList, ThisLayerQuery, LAYER_ISSUE_TIME
                )
                if this_clusterid not in used_cluster_id:
                    used_cluster_id.append(this_clusterid)
                """"""
                # run this q bulk
                LAYER_SUBMIT_TIME = LAYER_ISSUE_TIME
                ISSUE_T = LAYER_ISSUE_TIME
                SUBMIT_T = ISSUE_T
                SUBMIT_T_0 = ISSUE_T
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
                                _modelDAG.modelname,
                            )
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
                    # 1.3 run this_subquery
                    if isinstance(this_subquery, QR.QueryBuffer):
                        if this_subquery.execution == IF.BO.Read:
                            ISSUE_T = EXTRA_T
                            SUBMIT_T = self.ClusterList[
                                this_subquery.clusterid
                            ].run_query(this_subquery, ISSUE_T)
                            # SUBMIT_T = max(SUBMIT_T_0, SUBMIT_T) # for (R (REW))
                        if this_subquery.execution == IF.BO.Write:
                            ISSUE_T = SUBMIT_T
                            SUBMIT_T = self.ClusterList[
                                this_subquery.clusterid
                            ].run_query(this_subquery, ISSUE_T)
                    elif isinstance(this_subquery, QR.QueryExcute):
                        if isinstance(this_subquery, QR.QueryExcuteOnTC):
                            ISSUE_T = max(
                                self.ClusterList[this_clusterid].report_fb_submit_t(),
                                self.ClusterList[this_clusterid].report_wb_submit_t(),
                                SUBMIT_T,  # for VMM
                            )
                            SUBMIT_T = self.ClusterList[this_clusterid].run_query(
                                this_subquery, ISSUE_T
                            )
                        if isinstance(this_subquery, QR.QueryExcuteOnVPU):
                            if this_subquery.dfmod == "PostProcess":
                                this_clusterid = used_cluster_id[-1]
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
                                ISSUE_T = SUBMIT_T
                                SUBMIT_T_0 = self.ClusterList[
                                    this_subquery.clusterid
                                ].run_query(this_subquery, ISSUE_T)
                                SUBMIT_T = (
                                    SUBMIT_T_0
                                    if isinstance(
                                        subquerybulk[q_sub_idx + 1], QR.QueryExcuteOnVPU
                                    )
                                    else SUBMIT_T
                                )
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
                    LAYER_SUBMIT_TIME = max(LAYER_SUBMIT_TIME, SUBMIT_T)
            # END SUBQUERY
            # update layer submit time
            _modelDAG.update_submit_t(thisrun_index, LAYER_SUBMIT_TIME)
            # 2 hook3
            for h_idx in _modelDAG.fromVertexDict[thisrun_index]:
                if h_idx == -1:
                    continue
                _monitor.hook3_caf(
                    h_idx,
                    _modelDAG.LayerClassDict[thisrun_index],
                    _modelDAG.modelname,
                    self.ClusterList,
                )
            # 3 check end
            if not CandidateLayerList:
                break
        # END WHILE
        print("[Morax] Finish.")
        return


def schedule_a_cluster(
    _querylist: list,
    _clusterlist: list,
    _layerqueryclass: QR.LayerQuery,
    layer_issue_t,
) -> int:
    # TODO: Update greedy algo to a more advanced rule
    # ERR, NOT CONSIDER MEMORY SUBMIT TIME
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
    0520: Use VritualQuerySeparator to spcify bulk
    """
    # assert isinstance(_query_list[0], QR.QueryBuffer), "fatal, spciqury"
    # assert _query_list[0].execution == IF.BO.Read, "fatal, spciqury"
    this_querybulk = []
    while True:
        this_querybulk.append(copy.deepcopy(_query_list.pop(0)))
        if len(_query_list) == 0:
            break
        elif isinstance(_query_list[0], QR.VritualQuerySeparator):
            _query_list.pop(0)
            break

    """
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
    """
    return this_querybulk
