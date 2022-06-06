from sympy import assuming
import algorithm.offline as OFL
import morax.system.config as CFG
import morax.model.model as MD
import morax.hardware.chip as CP


def check_mapping_with_query(_layerindex):
    return True


class Mapper:
    def __init__(self) -> None:
        self.xbar_num = (
            CFG.MoraxConfig.ClusterNum
            * CFG.MoraxConfig.NVTCNum
            * CFG.MoraxConfig.RRAMSliceNum
            * CFG.MoraxConfig.RRAMXbarNum,
        )
        self.rram_cap_byte = (
            self.xbar_num
            * CFG.MoraxConfig.RRAMXbarSize
            * (CFG.MoraxConfig.RRAMXbarSize - CFG.MoraxConfig.RRAMLUTRows)
            * CFG.MoraxConfig.RRAMCellBits
        ) / 8
        self.xbar_size = (
            CFG.MoraxConfig.RRAMXbarSize,
            CFG.MoraxConfig.RRAMXbarSize - CFG.MoraxConfig.RRAMLUTRows,
        )
        self.bars_per_word = (
            CFG.MoraxConfig.PrecisionBits / CFG.MoraxConfig.RRAMCellBits
        )
        self.mapper_breakpoint = [0, 0, 0]  # clstid nvtcid sliceid

    def map(self, _modelDAG: MD.ModelDAG, _moraxChip: CP.MoraxChip):
        # _doclotnsl
        # dict of clstid: [tuple1(nvtcid, sliceidlist), tuple2, ... ]
        with assuming(self.bars_per_word == CFG.MoraxConfig.RRAMXbarNum):
            OnRRAMLayerIndexList = OFL.schedule_rram_layers(
                _modelDAG, self.xbar_num, self.xbar_size, self.bars_per_word
            )
            for orli in OnRRAMLayerIndexList:
                doclotnsl = {}
                xbar_num = OFL.calc_xbars(
                    _modelDAG.LayerClassDict[orli], self.xbar_size, self.bars_per_word
                )
                [clstid, nvtcid, sliceid] = self.mapper_breakpoint
                for xbars in range(0, xbar_num, self.bars_per_word):
                    # get_mapping_info
                    
                    _moraxChip.ClusterList[clstid].nvTensorCoreList[nvtcid]



