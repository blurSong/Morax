class MoraxConfig:
    # [Chip]
    OffChipBandwidthGbps = 256
    ClusterNum = 16

    # [Cluster]
    WeightBufferSizeKB = 1024
    FeatureBufferSizeKB = 256
    FeatureBufferBandwidthGbps = 64
    TCNum = 8
    nvTCNum = 8

    # [TensorCore]
    PEArrayNum = 8
    PEArrayBufferSizeKB = 64
    PEArraySize = 32
    NOCBwGbps = 1024

    # [nvTensorCore]
    RRAMSliceNum = 16
    RRAMXbarNum = 8
    RRAMXbarSize = 128
    RRAMLUTRows = 8
    RRAMSlicesTreeType = 1  # 1: Two-leaves Tree 2: H-tree 3：Fat-tree

    # [VPU]
    LaneNum = 16
    LaneSize = 64
    FIFODeepth = 8

    # [SMU]
    SMURegFileSizekB = 1
    # [System]
    PrecisionBits = 16
    FracBits = 5


class HWParam:
    ADCSpeedGbps = 1.28
