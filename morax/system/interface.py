# interface of system and hardware for morax simulator
# Morax means the mixing of CMOS and RRAM
# Author: Tao Song.
# Last modification: 0316

# For batch > 1:
# RRAM:onefeaturebyoonefeature
# OS: onechannelbyonechannel
# Morax uses token to control scratchpad buffer hierachy

from msilib.schema import Control
from enum import Enum
from morax.model.layer import LinearLayerType as LLT, NonlinearLayerType as NLT


class ClusterComponent(Enum):
    WeightBuffer = 0
    FeatureBuffer = 1
    SMU = 2
    VPU = 3
    TensorCore = 4
    nvTensorCore = 5
    OtherCluster = 6
    DRAM = 7


class SystemOperator(Enum):
    Transpose = 100
    LookUp = 101
    Truncation = 102
    HWNC2CHWN = 103
    CHWN2HWNC = 104
    VMAX = 105
    VNORM = 106


class BufferOperator(Enum):
    Write = 200
    Read = 201


class TransmissionOperator(Enum):
    ReadDRAM = 300
    ClusterTransfer = 301


class VirtualOperator(Enum):
    SystemStart = 400
    SystemStop = 401
    EditMemonitor = 402
    QuerySeprator = 403


CC = ClusterComponent
SO = SystemOperator
BO = BufferOperator
TO = TransmissionOperator
VO = VirtualOperator


MoraxExecutionDict = {
    CC.TensorCore: [
        LLT.Linear,
        LLT.CONV,
        LLT.DWCONV,
        LLT.Residual,
        LLT.Batchnorm,
        LLT.TRCONV,
        LLT.NGCONV,
        LLT.VMM,
        LLT.GEMM,
        LLT.MADD,
        LLT.Layernorm,
        # NLT.Pooling,
    ],
    CC.nvTensorCore: [
        LLT.Linear,
        LLT.CONV,
        LLT.TRCONV,
        LLT.NGCONV,
        LLT.VMM,
        LLT.GEMM,
        SO.LookUp,
    ],
    CC.SMU: [SO.Transpose, SO.Truncation, SO.HWNC2CHWN, SO.CHWN2HWNC],
    CC.VPU: [
        # LLT.Linear,
        # LLT.Residual,
        LLT.VDP,
        LLT.VADD,
        LLT.VMUL,
        LLT.VMM,
        LLT.Batchnorm,
        LLT.Layernorm,
        NLT.Pooling,
        SO.VMAX,
        SO.VNORM,
        # NLT.Softmax1D,
        # NLT.Softmax2D,
    ],
}
