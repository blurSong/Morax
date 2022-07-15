from sympy import assuming
import algorithm.offline as OFL
import morax.system.config as CFG
import morax.model.model as MD
import morax.hardware.chip as CP
import math
import copy


def check_mapping_with_query(_layerindex):
    return True


class Mapper:
    def __init__(self, _LUTEN=True) -> None:
        self.lut_row = CFG.MoraxConfig.RRAMLUTRows if _LUTEN else 0
        self.xbar_num = (
            CFG.MoraxConfig.ClusterNum
            * CFG.MoraxConfig.NVTCNum  # tile
            * CFG.MoraxConfig.RRAMSliceNum  # pe
            * CFG.MoraxConfig.RRAMXbarNum  # 8 xbar
        )
        self.rram_cap_byte = (
            self.xbar_num
            * CFG.MoraxConfig.RRAMXbarSize
            * (CFG.MoraxConfig.RRAMXbarSize - self.lut_row)
            * CFG.MoraxConfig.RRAMCellBits
            // 8
        )
        self.xbar_size = (
            CFG.MoraxConfig.RRAMXbarSize - self.lut_row,
            CFG.MoraxConfig.RRAMXbarSize,
        )
        self.bars_per_word = (
            CFG.MoraxConfig.PrecisionBits // CFG.MoraxConfig.RRAMCellBits
        )
        self.mapper_breakpoint = [0, 0, 0]  # clstid nvtcid sliceid

    def get_mapping_info(
        self, _modelname, layeridx, _row_l, _col_l, slice_idx, _COL_MAJOR=True
    ):
        # info: (modelname, layerid, (rowid, colid))
        # mvm_row, mvm_col
        row_par = math.ceil(_row_l * 1.0 / self.xbar_size[0])
        col_par = math.ceil(_col_l * 1.0 / self.xbar_size[0])
        if _COL_MAJOR:
            row_id = slice_idx % row_par
            col_id = slice_idx // row_par
        else:
            row_id = slice_idx % col_par
            col_id = slice_idx // col_par
        mvm_row = (
            self.xbar_size[0] if row_id < row_par - 1 else _row_l % self.xbar_size[0]
        )
        mvm_col = (
            self.xbar_size[1] if col_id < col_par - 1 else _col_l % self.xbar_size[1]
        )
        return (_modelname, layeridx, (row_id, col_id)), mvm_row, mvm_col

    def make_doclotnsl(self, _slice_num):
        doclotnsl = {}
        SLICELIST = []
        for lst in range(0, CFG.MoraxConfig.RRAMSliceNum):  # 0 - 15
            SLICELIST.append(lst)
        doclotnsl[self.mapper_breakpoint[0]] = []
        while _slice_num > 0:
            begin = self.mapper_breakpoint[2]
            end = (
                begin + _slice_num
                if (begin + _slice_num) <= CFG.MoraxConfig.RRAMSliceNum
                else CFG.MoraxConfig.RRAMSliceNum
            )
            snum = end - begin
            doclotnsl[self.mapper_breakpoint[0]].append(
                (self.mapper_breakpoint[1], tuple(copy.deepcopy(SLICELIST[begin:end])),)
            )
            _slice_num = _slice_num - snum
            self.mapper_breakpoint[2] = end % CFG.MoraxConfig.RRAMSliceNum
            if end == CFG.MoraxConfig.RRAMSliceNum:
                if (self.mapper_breakpoint[1] + 1) // CFG.MoraxConfig.NVTCNum > 0:
                    self.mapper_breakpoint[0] = self.mapper_breakpoint[0] + 1
                self.mapper_breakpoint[1] = (
                    self.mapper_breakpoint[1] + 1
                ) % CFG.MoraxConfig.NVTCNum
        return doclotnsl

    def map_multi(self, _modelDAG: MD.ModelDAG, _moraxChip: CP.MoraxChip):
        return

    def map_single(
        self,
        _modelDAG: MD.ModelDAG,
        _moraxChip: CP.MoraxChip,
        _strategy: OFL.Strategy,
        _batch,
    ):
        # _doclotnsl
        # dict of clstid: [(nvtcid, sliceidlist), tuple2, ... ]
        with assuming(self.bars_per_word == CFG.MoraxConfig.RRAMXbarNum):
            CschduleList, OnRRAMLayerIndexList = OFL.schedule_layers(
                _modelDAG,
                self.xbar_num,
                self.xbar_size,
                self.bars_per_word,
                _strategy,
                _batch,
            )
            for orli in OnRRAMLayerIndexList:
                xbar_num, row_l, col_l = OFL.calc_xbars(
                    _modelDAG.LayerClassDict[orli],
                    self.xbar_size,
                    self.bars_per_word,
                    True,
                )
                [clstid, nvtcid, sliceid] = self.mapper_breakpoint
                slice_num = xbar_num // self.bars_per_word
                for s_idx in range(0, slice_num):
                    mapping_info, mvm_row, mvm_col = self.get_mapping_info(
                        _modelDAG.modelname, orli, row_l, col_l, s_idx
                    )
                    if sliceid >= CFG.MoraxConfig.RRAMSliceNum:
                        sliceid = sliceid % CFG.MoraxConfig.RRAMSliceNum
                        nvtcid = nvtcid + 1
                        if nvtcid >= CFG.MoraxConfig.NVTCNum:
                            nvtcid = nvtcid % CFG.MoraxConfig.NVTCNum
                            clstid = clstid + 1
                    _moraxChip.ClusterList[clstid].nvTensorCoreList[nvtcid].map_a_slice(
                        sliceid, mapping_info, mvm_row, mvm_col, self.lut_row
                    )
                    sliceid = sliceid + 1
                doclotnsl = self.make_doclotnsl(slice_num)
                assert (
                    self.mapper_breakpoint[0] == clstid
                    and self.mapper_breakpoint[1] == nvtcid
                    and self.mapper_breakpoint[2] == sliceid
                )
                _modelDAG.assign_layer(orli, True, doclotnsl)
        return CschduleList, OnRRAMLayerIndexList
