# Timeflim class of Morax
# Morax means the mixing of CMOS and RRAM
# Author: Tao Song.
# Last modification: 0316

from typing import Container
import re
import numpy as np
import subprocess as SP
import multiprocessing as MP
import openpyxl
import math
import copy
from enum import Enum
from collections import UserList

t
int64 = np.int64


class TimeStamp:
    def __init__(self, _execution: Enum, _issue_t: int, _label: str) -> None:
        # self.fromComp = _from
        # self.toComp = _to
        self.execution = _execution
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


class TimeFilm(UserList):
    def __init__(self) -> None:
        super().__init__()

    def append_stamp(self, _stamp: TimeStamp):
        if self[-1].submit_t >= _stamp.issue_t:
            _stamp.issue_t = self[-1].submit_t + 1
            _stamp.submit_t = _stamp.issue_t + _stamp.span
        thisstamp = copy.deepcopy(_stamp)
        self.append(thisstamp)
