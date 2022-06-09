import math
import morax.model.layer as LYR


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


def schedule_rram_layers(_modelDAG, xbar_num, xbar_size: tuple, bars_per_word):
    # _rram_rramcap: data capacity of all rrams
    OnRRAMLayerIndexList = []
    return OnRRAMLayerIndexList
