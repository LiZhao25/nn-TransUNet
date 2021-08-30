import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, Dataset
import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.dataloading.dataset_loading import load_dataset, DataLoader3D, DataLoader2D, unpack_dataset


from torch.utils.data import TensorDataset, DataLoader

def get_ACDC_loader2d():
    t = "Task213_ACDC"
    p = join("/home/lz/nnu/nnUNet_preprocessed", t, "nnUNetData_plans_v2.1_2D_stage0")
    dataset = load_dataset(p)
    with open(join(join("/home/lz/nnu/nnUNet_preprocessed", t), "nnUNetPlansv2.1_plans_2D.pkl"), 'rb') as f:
        plans = pickle.load(f)

    unpack_dataset(p)
    dl2d = DataLoader2D(dataset, (256, 256), np.array([256, 256]).astype(int)[1:], 150,
                        oversample_foreground_percent=0.33)

    return dl2d

class acdcDataset(Dataset):
    def __init__(self, transform=None):
        self.dl2d=get_ACDC_loader2d()

class ACDC_DataModule(pl.LightningDataModule):
    def __init__(self,data_dir: str = './'):
        super().__init__()
        self.data_dir = data_dir
        #self.dataset=ACDC_dataset()
        self.seed=1234
        train_len=100
        val_len=20
        test_len=30
        self.trainset, self.valset, self.testset = random_split(
            self.dataset, lengths=[train_len, val_len, test_len], generator=torch.Generator().manual_seed(self.seed)
        )

    def train_dataloader(self):
            return DataLoader(self.trainset, batch_size=50, num_workers=16)

    def val_dataloader(self):
            return DataLoader(self.valset, batch_size=20, num_workers=16)

    def test_dataloader(self):
            return DataLoader(self.testset, batch_size=50, num_workers=16)