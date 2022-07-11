from calendar import c
import math
from tkinter.tix import Select
from xml.dom.minidom import Element
from anyio import current_time
import iniconfig
from enum import Enum
import copy
import numpy as np
from morax import MoraxChip

import morax.model.layer as LYR
from morax.model.model import ModelDAG
import morax.system.query as QR
import morax.system.interface as IF
from morax.system.config import MoraxConfig
from morax.hardware.chip import spcify_querybulk
from morax.hardware.tensorcore import TensorCore
from morax.hardware.vpu import VPU


class Strategy(Enum):
    random = 0
    fifo = 1
    greedy = 2
    dcp = 3
    layerwaver = 4


class SchduleDAG:
    def __init__(self, _modelDAG: ModelDAG) -> None:
        self.layernum = _modelDAG.layernum
        self.LayerIndexList = copy.deepcopy(_modelDAG.LayerIndexList)
        self.toVertexDict = copy.deepcopy(_modelDAG.toVertexDict)
        self.fromVertexDict = copy.deepcopy(_modelDAG.fromVertexDict)
        self.LayerClassDict = copy.deepcopy(_modelDAG.LayerClassDict)
        self.ConcatList = copy.deepcopy(_modelDAG.ConcatList)

        self.LayerQueryClassDict = {}
        self.LayerOnCMOSDict_MT = {}
        self.LayerOnCMOSDict_CT = {}

        self.SchduleList = {}


def calc_xbars(
    _layerclass: LYR.LinearLayer, _xbarsize: tuple, _bars_per_word=8, _returnRC=False
):
    """do not mod this"""
    layertype = _layerclass.layer_type
    row_l, col_l = 0, 0
    dup = _bars_per_word
    if layertype == LYR.LinearLayerType.Linear or LYR.LinearLayerType.VMM:
        row_l = _layerclass.row_dim
        col_l = _layerclass.col_dim
    elif (
        layertype == LYR.LinearLayerType.CONV or layertype == LYR.LinearLayerType.TRCONV
    ):
        row_l = _layerclass.in_channel * (_layerclass.kernel_size ** 2)
        col_l = _layerclass.out_channel
    elif layertype == LYR.LinearLayerType.NGCONV:
        # grpdict = _doclotnsl[grp]
        row_l = (
            _layerclass.in_channel * (_layerclass.kernel_size ** 2) / _layerclass.group
        )
        col_l = _layerclass.out_channel / _layerclass.group
        dup = dup * _layerclass.group
    elif layertype == LYR.LinearLayerType.GEMM:
        if _layerclass.input_indecies_tuple[0] == 0:
            row_l = _layerclass.k_dim
            col_l = _layerclass.m_dim
        else:
            row_l = _layerclass.k_dim
            col_l = _layerclass.n_dim
    else:
        row_l, col_l = 0, 0
    row_par = math.ceil(row_l * 1.0 / _xbarsize[0])
    col_par = math.ceil(col_l * 1.0 / _xbarsize[0])
    xbars = dup * row_par * col_par
    if _returnRC:
        return xbars, row_l, col_l
    else:
        return xbars


def calc_memsize_byte(_layerclass: LYR.LinearLayer, _bytes_per_word: int):
    layertype = _layerclass.layer_type
    if layertype == LYR.LinearLayerType.Linear or LYR.LinearLayerType.VMM:
        params = _layerclass.row_dim * _layerclass.col_dim
    elif (
        layertype == LYR.LinearLayerType.CONV or layertype == LYR.LinearLayerType.TRCONV
    ):
        params = (
            _layerclass.out_channel
            * _layerclass.in_channel
            * (_layerclass.kernel_size ** 2)
        )
    elif layertype == LYR.LinearLayerType.NGCONV:
        params = (
            _layerclass.in_channel
            * _layerclass.out_channel
            * (_layerclass.kernel_size ** 2)
            / _layerclass.group
        )
    elif layertype == LYR.LinearLayerType.DWCONV:
        params = _layerclass.channel * (_layerclass.kernel_size ** 2)
    elif layertype == LYR.LinearLayerType.GEMM:
        if _layerclass.input_indecies_tuple[0] == 0:
            params = _layerclass.k_dim * _layerclass.m_dim
        elif _layerclass.input_indecies_tuple[1] == 0:
            params = _layerclass.k_dim * _layerclass.n_dim
        else:
            params = 0
    else:
        params = 0
    _layerclass.memsizebyte = params * _bytes_per_word
    return _layerclass.memsizebyte


def calc_mem_time(_layerclass: LYR.LinearLayer, _bytes_per_word):
    memsizebyte = calc_memsize_byte(_layerclass, _bytes_per_word)
    return memsizebyte * 8 / MoraxConfig.OffChipBandwidthGbps


def calc_compute_time(_layerquery: QR.LayerQuery):
    """called after queried"""
    wb_tstamp = np.zeros(MoraxConfig.ClusterNum, dtype=int)
    fb_tstamp = np.zeros(MoraxConfig.ClusterNum, dtype=int)
    rb_tstamp = np.zeros(MoraxConfig.ClusterNum, dtype=int)
    vpu_tstamp = np.zeros(MoraxConfig.ClusterNum, dtype=int)
    tc_tstamp = np.zeros([MoraxConfig.ClusterNum, MoraxConfig.TCNum], dtype=int)
    # clst_tsstamp = [0] * MoraxConfig.ClusterNum
    # one TC has some pearray as a smallest unit
    demmy_TC = TensorCore(0)
    demmy_VPU = VPU()
    clst_idx, tc_idx = 0, 0
    subquerylist = copy.deepcopy(_layerquery.SubQueryList)
    while subquerylist:
        subquerybulk = spcify_querybulk(subquerylist)
        sqb_t = 0
        next_clst = False
        for subq in subquerybulk:
            if isinstance(subq, QR.QueryBuffer):
                if subq.execution == IF.BO.Read:
                    if subq.locationEnum == IF.ClusterComponent.FeatureBuffer:
                        runtime = (
                            subq.databulkclass.sizebyte
                            * 8
                            / MoraxConfig.BufferReadBandwidthGbps
                        )
                        fb_tstamp[clst_idx] = fb_tstamp[clst_idx] + runtime
                        sqb_t = max(fb_tstamp[clst_idx], sqb_t)
                    elif subq.locationEnum == IF.ClusterComponent.WeightBuffer:
                        runtime = (
                            subq.databulkclass.sizebyte
                            * 8
                            / MoraxConfig.BufferReadBandwidthGbps
                        )
                        wb_tstamp[clst_idx] = wb_tstamp[clst_idx] + runtime
                        sqb_t = max(wb_tstamp[clst_idx], sqb_t)
                elif subq.execution == IF.BO.Write:
                    runtime = (
                        subq.databulkclass.sizebyte
                        * 8
                        / MoraxConfig.BufferWriteBandwidthGbps
                    )
                    rb_tstamp[clst_idx] = max(rb_tstamp[clst_idx], sqb_t) + runtime
                    sqb_t = rb_tstamp[clst_idx]
            elif isinstance(subq, QR.QueryExcute):
                if isinstance(subq, QR.QueryExcuteOnTC):
                    runtime = demmy_TC.run_demmy_query(subq)
                    tc_tstamp[clst_idx][tc_idx] = (
                        max(tc_tstamp[clst_idx][tc_idx], sqb_t) + runtime
                    )
                    sqb_t = tc_tstamp[clst_idx][tc_idx]
                    tc_idx = tc_idx + 1
                elif isinstance(subq, QR.QueryExcuteOnVPU):
                    runtime = demmy_VPU.run_demmy_query(subq)
                    # if subq.dfmod == "PostProcess" or "SoftMAX":
                    vpu_tstamp[clst_idx] = max(vpu_tstamp[clst_idx], sqb_t) + runtime
                    sqb_t = vpu_tstamp[clst_idx]
                    next_clst = True
                else:
                    continue
            else:
                continue
        #
        if tc_idx >= MoraxConfig.TCNum:
            next_clst = True
            tc_idx = 0
        if next_clst:
            clst_idx = clst_idx + 1
            if clst_idx >= MoraxConfig.ClusterNum:
                clst_idx = 0
        return max(
            np.amax(wb_tstamp),
            np.amax(fb_tstamp),
            np.amax(rb_tstamp),
            np.amax(vpu_tstamp),
            np.amax(tc_tstamp),
        )


def schedule_rram_layers(
    _modelDAG: ModelDAG,
    _xbar_num,
    _xbar_size: tuple,
    _bars_per_word,
    _strategy=Strategy.greedy,
    _batch=1,
):
    # basic mapping unit: slice
    bytes_per_word = MoraxConfig.PrecisionBits / 8
    # generate full-cmos queries
    schduleDAG = SchduleDAG(_modelDAG)
    QR.generate_demmy_queries(schduleDAG, _batch)
    # get runtime info
    for lyridx in schduleDAG.LayerIndexList:
        # if schduleDAG.LayerClassDict.layer_type >= 0:
        if isinstance(schduleDAG.LayerClassDict[lyridx], LYR.LinearLayer):
            schduleDAG.LayerOnCMOSMTDict[lyridx] = calc_mem_time(
                schduleDAG.LayerClassDict[lyridx], bytes_per_word
            )
        else:
            assert schduleDAG.LayerClassDict.layer_type < 0
            schduleDAG.LayerOnCMOSDict_MT[lyridx] = 0
        schduleDAG.LayerOnCMOSDict_CT[lyridx] = calc_compute_time(
            schduleDAG.LayerQueryClassDict[lyridx]
        )
    # choose strategy and get cmos sch
    if _strategy == Strategy.greedy:
        CMOSSchduleList = offline_greedy(schduleDAG)
    elif _strategy == Strategy.layerwaver:
        CMOSSchduleList = offline_layerwaver(schduleDAG)
    # make rram mapping sch
    # todo
    return


def offline_greedy(schduleDAG: SchduleDAG, _bpw):
    SchduleList = [-1]
    CandidateLayerList = copy.deepcopy(schduleDAG.toVertexDict[-1])
    while CandidateLayerList:
        mt = schduleDAG.LayerOnCMOSDict_MT[CandidateLayerList[0]]
        thislyr_idx = CandidateLayerList[0]
        for cdidx in CandidateLayerList:
            if schduleDAG.LayerOnCMOSDict_MT[cdidx] < mt:
                mt = schduleDAG.LayerOnCMOSDict_MT[cdidx]
                thislyr_idx = cdidx
        SchduleList.append(thislyr_idx)
        CandidateLayerList.remove(thislyr_idx)
        schduleDAG.LayerQueryClassDict[thislyr_idx].FINISHED_FLAG = True
        for candiidx in schduleDAG.toVertexDict[thislyr_idx]:
            iscandidate = True
            for tmpfromidx in schduleDAG.fromVertexDict[candiidx]:
                if (
                    tmpfromidx != thislyr_idx
                    and not schduleDAG.LayerQueryClassDict[tmpfromidx].FINISHED_FLAG
                ):
                    iscandidate = False
                    break
            if iscandidate:
                CandidateLayerList.append(candiidx)
        """
        # greedy no need to update mt
        ct = schduleDAG.LayerOnCMOSDict_CT[thislyr_idx]
        for cdidx in CandidateLayerList:
            schduleDAG.LayerOnCMOSDict_MT[cdidx] = (
                schduleDAG.LayerOnCMOSDict_MT[cdidx] - ct
                if ct < schduleDAG.LayerOnCMOSDict_MT[cdidx]
                else 0
            )
        """
    return SchduleList


def offline_layerwaver(schduleDAG: SchduleDAG, _bpw):
    return


"""
# DEMICODE
def get_rram_capacity():
    return 0


def no_rram_jam(DAG, layer):
    return False


def check_dataflow_preference(layer):
    return True


def morax(layer):
    smu = 0
    total = 0 + smu
    return total


graph_delay_table = {}  # Record minCost of traversed {G: [(mem_left, delay), (), ...]}
schedule_delay_table = []  # Record schedule init "XXXXXX..."
INF = np.Infinity
mem = get_rram_capacity()
min_delay_glb = INF

# prune1 对于确定的子问题(G, mem)和已经确定的运行时间delay， 如果已有问题(G, mem')， mem' > mem & delay' < delay 则剪枝
# purne2 引入数据流偏好
# purne3 rram jam

# 定义增益吗，贪心划分图
# mem 访存增益

def recursive_pruning(DAG, schedule, mem):
    if DAG.is_empty():
        return 0
    else:
        candidate_layer_list = DAG.root()
        min_delay = INF
        for layer in candidate_layer_list:
            layer_mem = get_rram_capacity(layer)
            if (
                layer_mem < mem
                and check_dataflow_preference(layer)  # prune 1
                and no_rram_jam(schedule, layer)  # purne 2
            ):
                S = "R"
                mem_left = mem - layer_mem
            else:
                S = "C"
                mem_left = mem
            layer_delay = morax(schedule, layer, S)
            schedule_delay = schedule_delay_table[schedule].get_issue_time(layer)
            delay = layer_delay + schedule_delay
            schedule.update(layer.index, S)
            for sch in schedule_delay_table.same_sch(schedule):
                if delay > schedule_delay_table[sch].delay and mem_left <= graph_delay_table(DAG.pop(layer))
            cost = morax(schedule, layer) + recursive_pruning(DAG.pop(layer), mem_left)
    return

"""
"""
R = 'R'
C = 'C'
schedule = []
optSchedule = []
G_table = {}  # Record minCost of traversed {G: [(mem_left, cur_delay), (), ...]}


def check_g_reco(DAG, cur_delay, mem):
    for G_reco in G_table[DAG]:
        if G_reco.mem >= mem and G_reco.cur_delay < cur_delay:
            return False


def recursive_pruning(DAG, mem):
    if DAG.is_empty():
        return 0
    else:
        cur_schedule = schedule
        candidate_layer_list = DAG.root()
        minDelay = INF
        for layer in candidate_layer_list:
            layer_mem = get_rram_capacity(layer)
            if (
                layer_mem < mem
                and check_dataflow_preference(layer)  # prune 1
                and no_rram_jam(schedule, layer)  # prune 2
            ):
                schedule.update(layer, R)
                cur_delay = morax(schedule, layer)
                if check_g_reco(DAG.wo(layer), cur_delay, mem-layer_mem) is False:  # prune 3
                    continue
                delay = cur_delay + recursive_pruning(DAG.wo(layer), mem - layer_mem)
                if delay < minDelay:
                    minDelay = delay
                    optSchedule = schedule
            # case C
            schedule = cur_schedule
            schedule.update(layer, C)
            cur_delay = morax(schedule, layer) 
            if check_g_reco(DAG.wo(layer), cur_delay, mem-layer_mem) is False:  # purne 3
                continue
            delay = cur_delay + recursive_pruning(DAG.wo(layer), mem)
            if delay < minDelay:
                minDelay = delay
                optSchedule = schedule
            schedule.remove(layer)
    return minDelay

mapping_time_table = {}

def graph_dfs(mapping):
    if mapping_time_table.has(mapping):
        return mapping_time_table[mapping]
    schedule = default_layer_schedule(mapping.C)
    schedule.adjust_order_to_minimize_RC_idle()
    schedule.delay = morax(schedule)
    path = critical_path(schedule)
    for node in path:
        if C2R_positive(mapping, node):
            mapping_try = mapping
            mapping_try.C2R(node):
            
    return

def layerwaver_like_cmos_schedule(DAG):
    Schdeule = []
    MT, CT <- Morax(DAG, config) # MT: memory access time, CT: computation time
    update_dag(DAG, MT, CT) # 点权重为本层执行时间，边权重为toVertex的访存时间，不考虑cluster
    while not all_chdeuled():
        Select node i with min(MT[i] + CT[i])
        Schdeule <- DAG[i]
        update DAG->vertex_weight with Schdeule
    return Schdeule
"""

