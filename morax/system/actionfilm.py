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


class TimeStamp:
    def __init__(self, _to, _from, _issue_t, _label: str) -> None:
        self.fromComp = _from
        self.toComp = _to
        self.span = 0
        self.label = _label
        self.issue_t = _issue_t
        self.submit_t = _issue_t - 1

    def update_submit_t(self, _submit_t):
        self.submit_t = _submit_t
        self.span = _submit_t - self.issue_t

    def update_span(self, _span):
        self.submit_t = self.issue_t + _span
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
