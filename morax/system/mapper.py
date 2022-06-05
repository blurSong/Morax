import algorithm.offline as OFL
import morax.system.config as CFG
import morax.model.model as MD


def check_mapping_with_query(_layerindex):
    return True


class Mapper:
    def __init__(self) -> None:
        self.xbars = (
            CFG.MoraxConfig.NVTCNum
            * CFG.MoraxConfig.RRAMSliceNum
            * CFG.MoraxConfig.RRAMXbarNum
        )
        self.rramcap = (
            self.xbars
            * CFG.MoraxConfig.RRAMXbarSize
            * (CFG.MoraxConfig.RRAMXbarSize - CFG.MoraxConfig.RRAMLUTRows)
            * CFG.MoraxConfig.RRAMCellBits
        ) / CFG.MoraxConfig.PrecisionBits

    def map(self, _modelDAG: MD.ModelDAG):
        # assign_layer
        # _doclotnsl
        # dict of clstid: [tuple1(nvtcid, sliceidlist), tuple2, ... ]
        OnRRAMLayerIndexList = OFL.schedule_rram_layers(
            _modelDAG,
            self.xbars,
            (
                CFG.MoraxConfig.RRAMXbarSize,
                CFG.MoraxConfig.RRAMXbarSize - CFG.MoraxConfig.RRAMLUTRows,
            ),
        )
        for orli in OnRRAMLayerIndexList:
            

