import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.nn import CrossEntropyLoss
from torch import optim

from nnunet.TransUNet.utils import DiceLoss, test_single_volume
from nnunet.TransUNet.vit_seg_modeling import VisionTransformer as ViT_seg
from nnunet.TransUNet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.data_augmentation.default_data_augmentation import get_default_augmentation
from nnunet.training.dataloading.dataset_loading import load_dataset, DataLoader3D, DataLoader2D, unpack_dataset
from nnunet.TransUNet.utils import DiceLoss
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.tensor_utilities import sum_tensor

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

    # t = "Task213_ACDC"
    # p = join("/home/lz/nnu/nnUNet_preprocessed", t, "nnUNetData_plans_v2.1_2D_stage0")
    # dataset = load_dataset(p)
    # with open(join(join("/home/lz/nnu/nnUNet_preprocessed", t), "nnUNetPlansv2.1_plans_2D.pkl"), 'rb') as f:
    #     plans = pickle.load(f)
    #
    unpack_dataset(p)
    path = "/home/lz/Trans-nnUNet/ckpt/model_best_dice.pth"
    test_save_path = "/home/lz/Trans-nnUNet/pred"

    model.load_state_dict(torch.load(path))
    online_eval_foreground_dc = []
    online_eval_tp = []
    online_eval_fp = []
    online_eval_fn = []

    dl2d = DataLoader2D(dataset, (256, 256), np.array([256, 256]).astype(int)[1:], 1,
                        oversample_foreground_percent=0.33)
    tr, val = get_default_augmentation(dl2d, dl2d, np.array([256, 256]).astype(int))

    metric_list = np.zeros([3,2])

    model.eval()
    dice_hist = []

    for i in range(36):
        online_eval_foreground_dc = []
        online_eval_tp = []
        online_eval_fp = []
        online_eval_fn = []
        dict_tr = next(val)
        data, target = dict_tr['data'], dict_tr['target']
        data, target = data.cuda(), target.cuda()
        case_name = dict_tr["keys"]
        metric_i = test_single_volume(data, target, model, classes=4, patch_size=[256,256],
                                       case=case_name, z_spacing=1)
        pred = model(data)
        #loss_dice, dice_tr = dice_loss(pred, target.squeeze(1).long(), softmax=True)
        metric_list += np.array(metric_i)
        metric_list += np.array(metric_i)
        #print(metric_i)
        num_classes = pred.shape[1]
        output_softmax = softmax_helper(pred)
        output_seg = output_softmax.argmax(1)
        target = target[:, 0]
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

        online_eval_tp = np.sum(online_eval_tp, 0)
        online_eval_fp = np.sum(online_eval_fp, 0)
        online_eval_fn = np.sum(online_eval_fn, 0)

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in
                                           zip(online_eval_tp, online_eval_fp, online_eval_fn)]
                               if not np.isnan(i)]
        dice = [np.round(i, 4) for i in global_dc_per_class]
        print("Average global foreground Dice:", dice)
        dice_hist.append(np.mean(dice))


    mean_hd95 = np.mean(metric_list, axis=0)[1]#Hausdorff distance
    dice_mean = np.mean(dice_hist)#mean dice

    print(dice_hist)#dice similarity coefficient of each class
    print(dice_mean)
    print(mean_hd95)