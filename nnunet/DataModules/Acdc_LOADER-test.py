import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import *

from nnunet.training.data_augmentation.default_data_augmentation import get_default_augmentation
from nnunet.training.dataloading.dataset_loading import load_dataset, DataLoader3D, DataLoader2D, unpack_dataset

import os


from torch.utils.data import TensorDataset, DataLoader

from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.utilities.to_torch import maybe_to_torch


def ACDC_dataset():
    t = "Task213_ACDC"
    p = join("/home/lz/nnu/nnUNet_preprocessed", t, "nnUNetData_plans_v2.1_2D_stage0")
    dataset = load_dataset(p)
    with open(join(join("/home/lz/nnu/nnUNet_preprocessed", t), "nnUNetPlansv2.1_plans_2D.pkl"), 'rb') as f:
        plans = pickle.load(f)

    unpack_dataset(p)
    dl2d = DataLoader2D(dataset, (256, 256), np.array([256, 256]).astype(int)[1:], 150,
                        oversample_foreground_percent=0.33)
    batch = dl2d.generate_train_batch()
    print(batch["seg"].__class__)

    data = torch.Tensor(batch["data"])
    seg = torch.Tensor(batch["seg"])

    dataset = TensorDataset(data, seg)

    return dataset

if __name__ == '__main__':
    t = "Task213_ACDC"
    p = join("/home/lz/nnu/nnUNet_preprocessed", t, "nnUNetData_plans_v2.1_2D_stage0")
    dataset = load_dataset(p)
    with open(join(join("/home/lz/nnu/nnUNet_preprocessed", t), "nnUNetPlansv2.1_plans_2D.pkl"), 'rb') as f:
        plans = pickle.load(f)

    unpack_dataset(p)
    dl2d = DataLoader2D(dataset, (256, 256), np.array([256, 256]).astype(int)[1:], 16,
                        oversample_foreground_percent=0.33)
    tr, val = get_default_augmentation(dl2d, dl2d, np.array([256, 256]).astype(int))
    # for i in range():
    #     __ =next(tr)
    #     datict = next(tr)
    #     data = data_dict['data']
    #     target = data_dict['target']
    #     print(data_dict["keys"])

    trainer = nnUNetTrainerV2(fold=0, dataset_directory="/home/lz/nnu/nnUNet_preprocessed/Task213_ACDC", plans_file="/home/lz/nnu/nnUNet_preprocessed/Task213_ACDC/nnUNetPlansv2.1_plans_2D.pkl",
                              output_folder="/home/lz/Trans-nnUNet")

    trainer.initialize()

    tr_dl, val_dl = trainer.ret_dataloader()
    tr_gen, val_gen = get_default_augmentation(tr_dl, val_dl, np.array([256, 256]).astype(int))

    for i in range(1):
        __ =next(tr_gen)
        data_dict = next(tr_gen)
        data = data_dict['data']
        target = data_dict['target']
        #target = torch.Tensor(target)
        print(target.__class__)
        print(data_dict["keys"])

    #testdata = dl2d.get_item(0)
    #print(testdata["case_properties"])


    # path = "/home/lz/nnu/nnUNet_preprocessed/patient001_frame01.npy"
    # inst = np.load(path, "r")
    # print(inst.shape[1])
    #
    # datapaths = [f'/home/lz/nnu/nnUNet_preprocessed/']
    #
    # gt = join("/home/lz/nnu/nnUNet_preprocessed", t, "gt_segmentations")
    # niftilist = os.listdir(gt)
    # npylist = sorted([(item[0:-6]+ "npy") for item in niftilist])
    # print(npylist)
    # datamat = [np.load(join(p, item)) for item in npylist]
    #print(datamat)

