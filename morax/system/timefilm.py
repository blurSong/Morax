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

from sympy import im
from morax.system.interface import VO
from morax.system.config import MoraxConfig

int64 = np.int64


class TimeStamp:
    def __init__(self, _execution: Enum, _issue_t: int, _label: str) -> None:
        # self.fromComp = _from
        # self.toComp = _to
        self.execution = _execution
        self.span = 0
        self.label = _label
        self.issue_t = _issue_t
        self.submit_t = _issue_t

    def update_submit_t(self, _submit_t):
        self.submit_t = _submit_t
        self.span = _submit_t - self.issue_t

    def update_span(self, _span):
        self.submit_t = self.issue_t + _span
        self.span = _span


class TimeFilm(UserList):
    def __init__(self) -> None:
        super().__init__()
        self.append(TimeStamp(VO.SystemStart, 0, "SystemStart"))

    def append_stamp(self, _stamp: TimeStamp):
        if self[-1].submit_t >= _stamp.issue_t:
            _stamp.issue_t = self[-1].submit_t + 1
            _stamp.submit_t = _stamp.issue_t + _stamp.span
        this_stamp = copy.deepcopy(_stamp)
        self.append(this_stamp)

    def append_stamp_bufferver(self, _stamp: TimeStamp):
        # Added 0521 for OOO EXE
        if re.search("read", _stamp.label):
            for tidx in range(len(self) - 1, -1, -1):
                if re.search("write", self[tidx].label):
                    continue
                elif re.search("read", self[tidx].label):
                    if self[tidx].submit_t >= _stamp.issue_t:
                        _stamp.issue_t = self[tidx].submit_t + 1
                        _stamp.submit_t = _stamp.issue_t + _stamp.span
                        break
        elif re.search("write", _stamp.label):
            for tidx in range(len(self) - 1, -1, -1):
                if re.search("read", self[tidx].label):
                    continue
                elif re.search("write", self[tidx].label):
                    if self[tidx].submit_t >= _stamp.issue_t:
                        _stamp.issue_t = self[tidx].submit_t + 1
                        _stamp.submit_t = _stamp.issue_t + _stamp.span
                        break
        this_stamp = copy.deepcopy(_stamp)
        self.append(this_stamp)
        return
