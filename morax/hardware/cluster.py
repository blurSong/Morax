# Cluster class of Morax
# Morax means the mixing of CMOS and RRAM
# Author: Tao Song.
# Last modification: 0327

import numpy as np
import subprocess as SP
import multiprocessing as MP
import copy
import re
from morax.system.interface import BO, ClusterComponent
from morax.system.timefilm import TimeFilm, TimeStamp
from morax.system.config import MoraxConfig, HWParam
import morax.system.query as Q
import buffer
import nvtensorcore
import tensorcore
import smu
import vpu


class MoraxCluster:
    def __init__(self, _clusterid: int) -> None:
        self.clusterid = _clusterid
        self.WeightBuffer = buffer.ScratchpadBuffer(MoraxConfig.WeightBufferSizeKB, 0)
        self.FeatureBuffer = buffer.ScratchpadBuffer(
            MoraxConfig.FeatureBufferSizeKB, MoraxConfig.FeatureBufferBandwidthGbps
        )
        self.VPU = vpu.VPU()
        self.SMU = smu.SMU()
        self.TCNum = MoraxConfig.TCNum
        self.NVTCNum = MoraxConfig.NVTCNum
        self.TensorCoreList = []
        self.nvTensorCoreList = []
        for tcid in range(self.TCNum):
            tc = tensorcore.TensorCore(tcid)
            self.TensorCoreList.append(copy.deepcopy(tc))
        for nvtcid in range(self.NVTCNum):
            nvtc = nvtensorcore.nvTensorCore(nvtcid)
            self.nvTensorCoreList.append(copy.deepcopy(nvtc))
        self.ClusterTimeFilm = TimeFilm()

    def run_query(self, _q, _issue_t):
        this_query = copy.deepcopy(_q)
        ret_t = 0
        if isinstance(this_query, Q.QueryBuffer):
            bulklabel = this_query.databulkclass.label
            if re.search("WET", bulklabel):
                ret_t = self.WeightBuffer.run_query(this_query, _issue_t)
            elif re.search("FTR", bulklabel):
                ret_t = self.FeatureBuffer.run_query(this_query, _issue_t)
            else:
                raise AttributeError
            if this_query.execution == BO.Read and ret_t == -1:
                # TODO: Apply dram or inter-cluter query
                return _issue_t
        elif isinstance(this_query, Q.QueryExcuteOnNVTC):
            nvtcid = this_query.nvtcid
            ret_t = self.nvTensorCoreList[nvtcid].run_query(this_query, _issue_t)
        elif isinstance(this_query, Q.QueryExcuteOnTC):
            for tcid in range(self.TCNum):
                if self.TensorCoreList[tcid].TimeFilm[-1].submit_t < _issue_t:
                    ret_t = self.TensorCoreList[tcid].run_query(this_query, _issue_t)
            if ret_t == 0:  # all busy
                # TODO: Apply query on other cluter
                return _issue_t
        elif isinstance(this_query, Q.QueryExcuteOnVPU):
            if (
                self.VPU.TimeFilm[-1].submit_t < _issue_t
                or this_query.dfmod == "PostProcess"
            ):
                ret_t = self.VPU.run_query(this_query, _issue_t)
            else:
                # TODO: Apply query on other cluter
                return _issue_t
        elif isinstance(this_query, Q.QueryExcuteOnSMU):
            ret_t = self.SMU.run_query(this_query, _issue_t)
        return ret_t

