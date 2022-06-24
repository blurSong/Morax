from calendar import c
import math
from xml.dom.minidom import Element

import numpy as np
import morax.model.layer as LYR
from morax.model.model import ModelDAG
from enum import Enum


class Strategy(Enum):
    fifo = 0
    random = 1
    greedy = 2
    dp = 3


def calc_xbars(
    _layerclass: LYR.LinearLayer, _xbarsize: tuple, _bars_per_word=8, _returnRC=False
):
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
    row_par = math.ceil(row_l * 1.0 / _xbarsize[0])
    col_par = math.ceil(col_l * 1.0 / _xbarsize[0])
    xbars = dup * row_par * col_par
    if _returnRC:
        return xbars, row_l, col_l
    else:
        return xbars


def schedule_rram_layers(
    _modelDAG: ModelDAG,
    _xbar_num,
    _xbar_size: tuple,
    _bars_per_word,
    _strategy: Strategy,
):
    # basic mapping unit: slice
    total_slice_num = _xbar_num / _bars_per_word
    if _strategy == Strategy.greedy:
        OnRRAMLayerIndexList = offline_greedy(
            _modelDAG, total_slice_num, _xbar_size, _bars_per_word
        )
    elif _strategy == Strategy.dp:
        OnRRAMLayerIndexList = offline_dp(
            _modelDAG, total_slice_num, _xbar_size, _bars_per_word
        )
    return OnRRAMLayerIndexList


def offline_greedy(_modelDAG: ModelDAG, _slice, _xbar_size: tuple, _bpw):
    OnRRAMLayerIndexList = []
    return OnRRAMLayerIndexList


def offline_dp(_modelDAG: ModelDAG, _slice, _xbar_size: tuple, _bpw):
    OnRRAMLayerIndexList = []
    candidate_layer_list = [-1]
    # init
    for lyr in _modelDAG.LayerIndexList:
        _modelDAG.LayerAssignmentDict = []  # C

    while candidate_layer_list:
        for idx in candidate_layer_list:
            if idx == -1:
                candidate_layer_list = _modelDAG.LayerIndexList
    return OnRRAMLayerIndexList


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
# 

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

'''
'''
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

def default_layer_schedule():
    return

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
