import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.nn import CrossEntropyLoss
from torch import optim

from nnunet.TransUNet.utils import DiceLoss
from nnunet.TransUNet.vit_seg_modeling import VisionTransformer as ViT_seg
from nnunet.TransUNet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.data_augmentation.default_data_augmentation import get_default_augmentation
from nnunet.training.dataloading.dataset_loading import load_dataset, DataLoader3D, DataLoader2D, unpack_dataset
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.tensor_utilities import sum_tensor
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from time import time, sleep

if __name__ == "__main__":
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = 4
    config_vit.n_skip = 3
    model = ViT_seg(config_vit, img_size=256, num_classes=config_vit.n_classes).cuda()
    model.load_from(weights=np.load("/home/lz/pre-trained/R50+ViT-B_16.npz"))

    t = "Task213_ACDC"
    p = join("/home/lz/nnu/nnUNet_preprocessed", t, "nnUNetData_plans_v2.1_2D_stage0")
    dataset = load_dataset(p)
    with open(join(join("/home/lz/nnu/nnUNet_preprocessed", t), "nnUNetPlansv2.1_plans_2D.pkl"), 'rb') as f:
        plans = pickle.load(f)

    unpack_dataset(p)
    path = "/home/lz/Trans-nnUNet/ckpt"
    dl2d = DataLoader2D(dataset, (256, 256), np.array([256, 256]).astype(int)[1:], 36,
                        oversample_foreground_percent=0.33, pad_mode="constant", memmap_mode='r')
    trainer = nnUNetTrainerV2(fold=0, dataset_directory="/home/lz/nnu/nnUNet_preprocessed/Task213_ACDC",
                              plans_file="/home/lz/nnu/nnUNet_preprocessed/Task213_ACDC/nnUNetPlansv2.1_plans_2D.pkl",
                              output_folder="/home/lz/Trans-nnUNet")

    trainer.initialize()

    tr_dl, val_dl = trainer.ret_dataloader()
    tr_gen, val_gen = get_default_augmentation(tr_dl, val_dl, np.array([256, 256]).astype(int))
    # tr_gen, val_gen = trainer.ret_generator()

    loss_func = DC_and_CE_loss({'batch_dice': False, 'smooth': 1e-5, 'do_bg': False}, {})

    # loss_func = trainer.ret_loss()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    init_lr = 0.01
    optimizer = optim.SGD(model.parameters(), lr=init_lr, momentum=0.99, weight_decay=0.0001, nesterov=True)
    loss = []
    val_loss = []
    dice_tr = []
    dice_val = []

    tr_hist = []
    val_hist = []
    dice_hist = []

    online_eval_foreground_dc = []
    online_eval_tp = []
    online_eval_fp = []
    online_eval_fn = []

    for epoch in range(500):
        model.train()
        print("epoch: ", epoch+1)
        epoch_start_time = time()
        train_losses_epoch = []
        val_losses_epoch = []
        for batch in range(250):
            data_dict = next(tr_gen)
            data = data_dict['data']
            target = data_dict['target']
            # data = maybe_to_torch(data)
            # target = maybe_to_torch(target)
            data, target = data.cuda(), target.cuda()
            # data = to_cuda(data)
            # target = to_cuda(target)

            output = model(data)
            l = loss_func(output, target)
            l_cpu = l.detach().cpu().numpy()

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_losses_epoch.append(l_cpu)
        lr_ = init_lr * (1.0 - epoch / 1000) ** 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        with torch.no_grad():
            model.eval()
            for _ in range(50):
                dict_val = next(val_gen)
                val_data, val_target = dict_val['data'], dict_val['target']
                # val_data = maybe_to_torch(val_data)
                # val_target = maybe_to_torch(val_target)
                # val_data = to_cuda(val_data)
                # val_target = to_cuda(val_target)
                val_data, val_target = val_data.cuda(), val_target.cuda()
                val_output = model(val_data)
                l_val = loss_func(val_output, val_target)
                l_val_cpu = l_val.detach().cpu().numpy()
                val_losses_epoch.append(l_val_cpu)

                num_classes = val_output.shape[1]
                output_softmax = softmax_helper(val_output)
                output_seg = output_softmax.argmax(1)
                target = val_target[:, 0]
                axes = tuple(range(1, len(target.shape)))
                tp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
                fp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
                fn_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
                for c in range(1, num_classes):
                    tp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target == c).float(), axes=axes)
                    fp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target != c).float(), axes=axes)
                    fn_hard[:, c - 1] = sum_tensor((output_seg != c).float() * (target == c).float(), axes=axes)

                tp_hard = tp_hard.sum(0, keepdim=False).detach().cpu().numpy()
                fp_hard = fp_hard.sum(0, keepdim=False).detach().cpu().numpy()
                fn_hard = fn_hard.sum(0, keepdim=False).detach().cpu().numpy()

                online_eval_foreground_dc.append(list((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8)))
                online_eval_tp.append(list(tp_hard))
                online_eval_fp.append(list(fp_hard))
                online_eval_fn.append(list(fn_hard))

        tr_hist.append(np.mean(train_losses_epoch))
        val_hist.append(np.mean(val_losses_epoch))
        print("train loss: ", tr_hist[-1])
        print("validation loss: ", val_hist[-1])

        online_eval_tp = np.sum(online_eval_tp, 0)
        online_eval_fp = np.sum(online_eval_fp, 0)
        online_eval_fn = np.sum(online_eval_fn, 0)

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in
                                           zip(online_eval_tp, online_eval_fp, online_eval_fn)]
                               if not np.isnan(i)]
        dice = [np.round(i, 4) for i in global_dc_per_class]
        print("Average global foreground Dice:", dice)
        dice_hist.append(np.mean(dice))
        epoch_end_time = time()
        epoch_time = epoch_end_time - epoch_start_time
        print("This epoch tooks ", epoch_time, "s")

        online_eval_foreground_dc = []
        online_eval_tp = []
        online_eval_fp = []
        online_eval_fn = []

        if val_hist[-1] <= min(val_hist):
            if isfile(join(path, "model_best_loss.pth")):
                os.remove(join(path, "model_best_loss.pth"))
            save_path = join(path, "model_best_loss.pth")
            torch.save(model.state_dict(), save_path)
            print("loss checkpoint updated!")

        if dice_hist[-1] >= max(dice_hist):
            if isfile(join(path, "model_best_dice.pth")):
                os.remove(join(path, "model_best_dice.pth"))
            save_path = join(path, "model_best_dice.pth")
            torch.save(model.state_dict(), save_path)
            print("dice checkpoint updated!")

        print("\n")

