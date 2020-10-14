from torchvision import datasets, transforms
from data.datasets import SOL_ordinario
from base import BaseDataLoader, sol_collate, SoLBase
from utils import init_transform


class SOL_Dataloader(SoLBase):
    def __init__(self, path_to_data, batch_size, shuffle=True, validation_split=0.0, num_workers=1, split=0, **kwargs):
        if 'transform' in kwargs:
            kwargs['transform'] = init_transform(kwargs['transform'])
        self.dataset = SOL_ordinario(path_to_data, **kwargs)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, split)
