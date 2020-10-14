import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)


class SoLBase(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, split=0, collate_fn=default_collate):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        if split == 0:
            seed = 1111
        elif split in [1, 2, 3, 4, 5]:
            np.random.seed(1111)
            seeds = np.random.choice(9999, size=5)
            seed = seeds[split - 1]
        self.seed = seed


        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
        }

        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        dp = np.array(self.dataset.path_to_audio)
        inst = [str(i).split('/')[-1].split('-')[0] for i in dp]
        dict_inst_idx = {}
        inst_ratio = {}
        for i in sorted(set(inst)):
            inst_idx = np.where(np.array(inst) == i)[0]
            inst_ratio[i] = len(inst_idx) / len(dp)
            dict_inst_idx[i] = inst_idx

        dict_cat_inst_idx = {}
        dict_cat_inst_ratio = {}
        for k, v in self.dataset.instrument_map.items():
            try:
                dict_cat_inst_idx[v] = np.hstack([dict_cat_inst_idx[v], dict_inst_idx[k]])
                dict_cat_inst_ratio[v] += inst_ratio[k]
            except:
                dict_cat_inst_idx[v] = dict_inst_idx[k]
                dict_cat_inst_ratio[v] = inst_ratio[k]
        self.dataset.dict_cat_inst_idx = dict_cat_inst_idx
        self.dataset.dict_cat_inst_ratio = dict_cat_inst_ratio

        valid_idx = []
        for i in inst_ratio:
            np.random.seed(self.seed)
            valid_idx.append(np.random.choice(dict_inst_idx[i], size=int(split * len(dict_inst_idx[i])), replace=False))
        valid_idx = np.array([j for i in valid_idx for j in i])

        train_idx = np.setdiff1d(idx_full, valid_idx)
        self.valid_idx = valid_idx
        self.train_idx = train_idx

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)



def sol_collate(batch):
    """
    NOTE:
        when input x is waveform:
            x.dim() = 1
        when inupt x is spectrogram extracted by nnAudio:
            x.dim() = 3
            x.shape = (batch, time, freq)  

    Note in the second case, the last two dim is swapped (so must include Transpose() in Transform), 
    necessary for rnn.pad_sequence to work. Also, the batch dimension has be to removed.
    """
    list_xlen = []
    list_x = []
    list_idx = []
    list_y = []
    for x, idx, y in batch:
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)    

        # assuming x is the result of ExtractSpectrogram
        # (1, freq, time)
        x = x.transpose(1, 2).contiguous().view(-1, x.shape[1]).unsqueeze(-1)

        list_xlen.append(x.shape[0]) 
        list_x.append(x)
        list_idx.append(idx)
        list_y.append(y)

    x = torch.nn.utils.rnn.pad_sequence(list_x, batch_first=True, padding_value=0)
    # trailing_dims = list_x[0].size()[1:]
    # out_dims = (len(list_x), 162) + trailing_dims
    # x = torch.zeros(*out_dims).to(x.device)
    # for i, tensor in enumerate(list_x):
    #     length = tensor.size(0)
    #     x[i, :length, ...] = tensor

    xlen = torch.LongTensor(list_xlen).to(x.device)
    idx = torch.LongTensor(list_idx).to(x.device)
    y = torch.LongTensor(list_y).to(x.device)

    mask = len_to_mask(xlen, dtype=x.type())

    return x, xlen, mask, idx, y


def len_to_mask(length, dtype=None):
    assert len(length.shape) == 1, "length shape must be one, found {} instead.".format(length.shape)
    max_len = length.max().item()
    mask = torch.arange(max_len, device=length.device).expand(len(length), max_len) < length.unsqueeze(-1)
    if dtype is not None:
        mask = mask.type(dtype)

    return mask
