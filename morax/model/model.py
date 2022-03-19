import copy
import re
import sys
import os
from enum import Enum
from typing import Dict, List
import numpy as np
import pandas as pd
from pyrsistent import T
import torch
import torch.nn as nn
import torch.nn.functional as Func

from layer import *


class ModelDAG(Dict):
    def __init__(self, _modelname, _modeltype) -> None:
        super().__init__()
        self.model_name = _modelname
        self.model_type = _modeltype
        self.layernum = 0
        self.linearlayernum = 0
        self.nonlinearlayernum = 0

    def add_layer(self, _layerindex, _islinear):
        self.layernum += 1
        self.linearlayernum += 1 if _islinear is True else 0
        self.nonlinearlayernum += 1 if _islinear is False else 0
        self[_layerindex] = []

    def add_edge(self, _from, _to):
        self[_from].append(_to)


class LayerList(List):
    def __init__(self, _modelname, _modeltype) -> None:
        super().__init_()
        self.model_name = _modelname
        self.model_type = _modeltype
        self.layernum = 0

    def add_layer(self, _layerclass):
        super().append(_layerclass)
        self.layernum += 1
