import json
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict

import numpy as np 
import torch
from torchvision import transforms
import data.audio_processor as module_ap


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader): 
    ''' wrapper function for endless data loader. ''' 
    for loader in repeat(data_loader): 
        yield from loader 

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()
        
    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        # if self.writer is not None:
        #     self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]
    
    def result(self):
        return dict(self._data.average)


def set_seed(seed=123):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def init_transform(config_transform):
    """
    This should be called inside a `DataLoader`,
    used to initiate transformers for a `Dataset`.
    """
    compiled_transform = transforms.Compose([getattr(module_ap, k)(**v) for k, v in config_transform.items()])
    return compiled_transform
    
 
