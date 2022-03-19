# TensorCore class of Morax
# Morax means the mixing of CMOS and RRAM
# Author: Tao Song.
# Last modification: 0316

import sys
from typing import Container
import re
import numpy as np
import subprocess as SP
import multiprocessing as MP
import openpyxl
import math
from timefilm import TimeFilm
from morax.model.layer import LinearLayerType


class PEArray():
    def __init__(self, _height, _width, _bufsize) -> None:
        self.height = _height
        self.width = _width
        self.buffer_size = _bufsize
        self.row_bufline_size = _width
        self.timefilm = TimeFilm()
        self.busy = False

    def run_query(self, _dfmodel, _querytype, _queryargsdict):
        assert _dfmodel == 'os' or _dfmodel == 'sys'
        if _querytype == LinearLayerType.CONV:
            sub_fm_height = _queryargsdict['sub_fm_height']
            sub_fm_width = _queryargsdict['sub_fm_width']
