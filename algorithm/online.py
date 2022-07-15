from morax.model.model import ModelDAG
import math
import copy


# demicode
class Schduler:
    def __init__(self) -> None:
        self.OnRRAMLayerIndexList = []
        self.CschduleList = []

    def update_sch(self, _CschduleList, _OnRRAMLayerIndexList):
        self.OnRRAMLayerIndexList = copy.deepcopy(_OnRRAMLayerIndexList)
        self.CschduleList = copy.deepcopy(_CschduleList)

    def schedule_one_layer(
        self, _candidatelayerlist: list, _modelDAG: ModelDAG,
    ):
        # idx
        thisrun_index = -1
        sch_pos = len(self.CschduleList)
        for cand_idx in _candidatelayerlist:
            if cand_idx in self.OnRRAMLayerIndexList:
                thisrun_index = cand_idx
                break
            cand_pos = self.CschduleList.index(cand_idx)
            if cand_pos < sch_pos:
                thisrun_index = cand_idx
                sch_pos = cand_pos
        # ist
        issue_t = 0
        if thisrun_index in self.OnRRAMLayerIndexList:
            for fvec in _modelDAG.fromVertexDict[thisrun_index]:
                issue_t = max(issue_t, _modelDAG.SubmitTimeDict[fvec])
        else:
            for sched_vec in self.CschduleList[:sch_pos]:
                issue_t = max(issue_t, _modelDAG.SubmitTimeDict[sched_vec])
        #
        _modelDAG.update_issue_t(thisrun_index, issue_t + 1)
        return thisrun_index, issue_t + 1
