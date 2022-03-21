# Timeflim class of Morax
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
import copy
from enum import Enum
from typing import List

#  WB <-> TC
#  FB
#


class TimeStamp():
    def __init__(self, _to, _from, _issuetime) -> None:
        self.fromC = _from
        self.toC = _to
        self.span = 0
        self.issuetime = _issuetime
        self.submittime = _issuetime - 1

    def update_submittime(self, _submittime):
        self.submittime = _submittime
        self.span = _submittime - self.issuetime

    def update_span(self, _span):
        self.submittime = self.issuetime + _span
        self.span = _span


class TimeFilm(List):
    def __init__(self) -> None:
        super().__init__()

    def add_stamp(self, _stamp):
        if self[-1].submittime >= _stamp.issuetime:
            thisstamp = copy.deepcopy(_stamp)
            thisstamp.issuetime = self[-1].submittime + 1
            thisstamp.submittime = thisstamp.issuetime + thisstamp.span
            self.append(thisstamp)
