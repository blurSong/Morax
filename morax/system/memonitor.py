from morax.hardware.buffer import DataBulk
from morax.system.config import MoraxConfig, HWParam
import copy

def edit_data_index(_thisidx, _thatidx, datatype):
    # for concat or append

def check_scratchpad():
    return


class Scratchpad():
    # 1 改concat的index
    # 2 改卷积和矩阵乘的数据标签
    def __init__(self):
        self.Scratchpad = {}
        return
    
    def writeNote(self, _bulk: DataBulk):
        note = _bulk.modelname + '_' + str(_bulk.layerindex) + '_' + _bulk.datatype
        if note in self.Scratchpad:
            self.Scratchpad[note]['datasize'] += _bulk.bulksizebyte
            self.Scratchpad[note]['bulklabel'].append(_bulk.bulklabel)
            assert self.Scratchpad[note]['token'] == _bulk.token
        else:
            pad = {}
            pad['datasize'] = _bulk.bulksizebyte
            pad['bulklabel'] = [_bulk.bulklabel]
            pad['token'] = _bulk.token
            self.Scratchpad[note] = copy.deepcopy(pad)

    def readNote(self, _bulk: DataBulk):
        note = _bulk.modelname + '_' + str(_bulk.layerindex) + '_' + _bulk.datatype
        if note in self.Scratchpad:
            self.Scratchpad[note]['datasize'] -= _bulk.bulksizebyte
            return "Successed"
        else:
            return 'Failed'
    
    def delNote(self, _note):
        assert _note in self.Scratchpad
        del self.Scratchpad[_note]
    
    def clearScratchpad(self):
        self.Scratchpad.clear()


class MemMonitor():
    def __init__(self) -> None:
        self.scratchpadnum = MoraxConfig.ClusterNum
        self.monitor = {}

    def insert_note(self, _note: str, _size: int, _location: int):
        self.monitor[_note] = {}
        self.monitor[_note]['size'] = _size
        self.monitor[_note]['loclist'] = [_location]
        assert _location in range(0, self.scratchpadnum) # -1 for offchip
    
    def transfer_note(self, _note, _from, _to):
        assert _from in self.monitor[_note]['loclist']
        self.monitor[_note]['loclist'].append(_to)

    def eliminate_note(self, _note, _location):
        self.monitor[_note]['loclist'].remove(_location)
        if not self.monitor[_note]['loclist']:
            del self.monitor[_note]
    
    def search_note(self, _note):
        if _note in self.monitor:
            return self.monitor[_note]['size'], self.monitor[_note]['loclist']
        else:
            return 0, []

    def edit_note(self, _note, _newnote):
        self.monitor[_newnote] = copy.deepcopy(_note)
        del self.monitor[_note]

    
    










        