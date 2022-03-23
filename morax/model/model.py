import copy
import re
import sys
import os
from enum import Enum
from typing import Dict, List
from pyrsistent import T
import morax.model.layer as Lyr


class ModelType(Enum):
    MLP = 0
    CNN = 1
    RNN = 2
    LSTM = 3
    MHATTENTION = 4


class ModelDAG(Dict):
    def __init__(self, _modelname, _modeltype) -> None:
        super().__init__()
        self.modelname = _modelname
        self.modeltype = _modeltype
        self.layernum = 0
        self.linearlayernum = 0
        self.nonlinearlayernum = 0

    def add_layer(self, _layerindex: int, _islinear):
        self.layernum += 1
        self.linearlayernum += 1 if _islinear is True else 0
        self.nonlinearlayernum += 1 if _islinear is False else 0
        self[_layerindex] = []

    def add_edge(self, _from, _to):
        self[_from].append(_to)

    def add_vlayer(self):
        self.add_layer(-1, False)


class ModelList(List):
    def __init__(self, _modelname, _modeltype) -> None:
        super().__init_()
        self.modelname = _modelname
        self.modeltype = _modeltype
        self.layernum = 0

    def add_layer(self, _layerclass):
        super().append(_layerclass)
        self.layernum += 1

    def add_vlayer(self):
        vlayerclass = Lyr.Layer("begin", -1)
        super().insert(0, vlayerclass)

