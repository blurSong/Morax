import copy
from enum import Enum
from collections import UserDict, UserList
import morax.model.layer as Lyr


class ModelType(Enum):
    MLP = 0
    CNN = 1
    RNN = 2
    LSTM = 3
    MHATTENTION = 4


CNNModelList = [
    "alexnet",
    "vgg16",
    "vgg19",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnext50",
    "mobilenet_v2",
    "shufflenet_v2",
    "unet",
]

MLPModelList = []

AttentionModelList = []


class ModelDAG:
    def __init__(self, _modelname, _modeltype) -> None:
        self.modelname = _modelname
        self.modeltype = _modeltype
        self.layernum = 0
        self.linearlayernum = 0
        self.nonlinearlayernum = 0

        self.LayerIndexList = []
        self.toVertexDict = {}
        self.fromVertexDict = {}

        self.LayerClassDict = {}
        self.LayerAssignmentDict = {}
        self.LayerQueryClassDict = {}

        self.ConcatList = []

        self.assigned = False
        self.compiled = False

    def add_layer(self, _layerindex: int, _islinear):
        self.layernum += 1
        self.linearlayernum += 1 if _islinear is True else 0
        self.nonlinearlayernum += 1 if _islinear is False else 0

        self.LayerIndexList.append(_layerindex)
        self.toVertexDict[_layerindex] = []
        self.fromVertexDict[_layerindex] = []

    def add_edge(self, _from, _to):
        self.toVertexDict[_from].append(_to)
        self.fromVertexDict[_to].append(_from)

    def assign_layer(
        self,
        _layerindex,
        _onRRAM=False,
        _doclotnsl: dict = {},
        # dict of clstid: [tuple1(nvtcid, sliceidlist), tuple2, ... ]
    ):
        # NOTE assume one cluster is engouh for one layer now
        if _onRRAM:
            doclotnsl = copy.deepcopy(_doclotnsl)
            self.LayerAssignmentDict[_layerindex] = doclotnsl
        else:
            self.LayerAssignmentDict[_layerindex] = {}
        return

    def add_vlayer(self):
        # self.add_layer(-1, False)
        self.toVertexDict[-1] = []
        self.fromVertexDict[-1] = []


class ModelList(UserList):
    def __init__(self, _modelname, _modeltype) -> None:
        super().__init__()
        self.modelname = _modelname
        self.modeltype = _modeltype
        self.layernum = 0

    def add_layer(self, _layerclass):
        super().append(_layerclass)
        self.layernum += 1

    def add_vlayer(self):
        vlayerclass = Lyr.Layer("begin", -1)
        super().insert(0, vlayerclass)

