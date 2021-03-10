import torch as t 
import h5py as hpy 
import pandas as pd 
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad
from prefetch_generator import BackgroundGenerator
from pytorch_lightning import LightningDataModule
import os 

    
class CollateFn():
    def __init__(self) -> None:
        self.bos_id = 2.0
        self.eos_id = 3.0
    
    def __call__(self, batch):
        feature, feature_length, target, target_length = [],[],[],[]
        for i in batch:
            feature.append(i[0])
            feature_length.append(i[1])
            target.append(i[2])
            target_length.append(i[3])
        feature = pad_sequence(feature, batch_first=True)
        feature_length = t.cat(feature_length)
        ctc_target = pad_sequence(target, batch_first=True)
        ctc_target_length = t.cat(target_length)
        att_input = pad(ctc_target, (1, 0), value=self.bos_id)
        att_output = pad(ctc_target, (0, 1), value=0.0)
        att_output = att_output.scatter(1, ctc_target_length.unsqueeze(-1), self.eos_id)
        # ctc_target = t.LongTensor([0,1,2,10,11,12,13,4209,4210,4211,4212, 0])
        ctc_lan_target = ctc_target.masked_fill(ctc_target.ge(4210), 3.0)
        ctc_lan_target = ctc_lan_target.masked_fill((ctc_target.ge(11) & ctc_target.lt(4210)), 2.0)
        ctc_lan_target = ctc_lan_target.masked_fill((ctc_target.gt(0) & ctc_target.lt(11)), 1.0)
        att_lan_target = pad(ctc_lan_target, (1, 0), value=1.0)
        return feature, feature_length, ctc_target, att_input, att_output, ctc_target_length, ctc_lan_target, att_lan_target

class HdfDataSet(Dataset):
    def __init__(self, hdf_file):
        super(HdfDataSet, self).__init__()
        self.hdf_file = hdf_file
        with hpy.File(hdf_file, 'r') as reader:
            self.length = len(reader['dataset'])

    def init_file(self):
        self.reader = hpy.File(self.hdf_file, 'r')
        self.dataset = self.reader['dataset']

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if not hasattr(self, 'dataset'):
            self.init_file()
        feature, feature_length, feature_size, target = self.dataset[index]
        feature = t.from_numpy(feature.reshape(feature_length, feature_size)).view(feature_length, feature_size)
        feature_length = t.LongTensor([feature_length])
        target = t.LongTensor(target)
        target_length = t.LongTensor([len(target)])
        return feature, feature_length, target, target_length

class FasterDataloader(DataLoader):
    def __init__(self, max_prefetch, *args, **kwargs):
        super(FasterDataloader, self).__init__(*args, **kwargs)
        # self.collate_fn = FasterCollateFn()
        self.max_prefetch = max_prefetch

    def __iter__(self):
        return BackgroundGenerator(super().__iter__(), max_prefetch=self.max_prefetch)

class DataModule(LightningDataModule):
    def __init__(self):
        super(DataModule, self).__init__()
        self.hdf_root = 'data/hdf'
        self.train_files = [
            'data_aishell_train.hdf', 'data_aishell_dev.hdf',
            'aidatatang_200zh.hdf', 'AISHELL-2.hdf', 'magic_data_train.hdf', 'magic_data_dev.hdf', 
            'prime.hdf', 'stcmds.hdf', 
            # 'libri_360.hdf', 'libri_500.hdf',
            # 'ce_200.hdf', '90_ce_200.hdf', '95_ce_200.hdf','105_ce_200.hdf','110_ce_200.hdf',
            # 'libri_360.hdf', 'libri_500.hdf'
        ]
        self.train_files = [os.path.join(self.hdf_root, i) for i in self.train_files]
        self.dev_files = [
            'data_aishell_test.hdf',
            'magic_data_test.hdf', 
            # 'ce_20_dev.hdf',
            # 'libri_100.hdf', 
        ]
        self.dev_files = [os.path.join(self.hdf_root, i) for i in self.dev_files]

    def prepare_data(self, *args, **kwargs):
        pass

    def train_dataloader(self, batch_size=64, max_prefetch=5):
        dataset = ConcatDataset([HdfDataSet(i) for i in self.train_files])
        dataloader = FasterDataloader(max_prefetch=max_prefetch, pin_memory=True, dataset=dataset, shuffle=False, batch_size=batch_size, num_workers=8, collate_fn=CollateFn())
        return dataloader

    def val_dataloader(self, batch_size=64, max_prefetch=5):
        dataset = ConcatDataset([HdfDataSet(i) for i in self.dev_files])
        dataloader = FasterDataloader(max_prefetch=max_prefetch, dataset=dataset, batch_size=batch_size, num_workers=8, collate_fn=CollateFn())
        return dataloader



# if __name__ == '__main__':
#     hdf_file = 'data/hdf/data_aishell_train.hdf'
#     dataset = HdfDataSet(hdf_file)
#     collate_fn = CollateFn()
#     batch = []
#     for i in range(32):
#         batch.append(dataset[i])