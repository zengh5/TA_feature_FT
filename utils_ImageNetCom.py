import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import os
import scipy.stats as st
import csv


# simple Module to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]


# load image metadata (Image_ID, true label, and target label)
def load_ground_truth(csv_filename):
    image_id_list = []
    label_ori_list = []
    label_tar_list = []

    with open(csv_filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            image_id_list.append(row['ImageId'])
            label_ori_list.append(int(row['TrueLabel']) - 1)
            label_tar_list.append(int(row['TargetClass']) - 1)

    return image_id_list, label_ori_list, label_tar_list


def FIAloss(grad, feature):
    Loss = torch.sum(grad * feature)   # 这才是论文
    return Loss


# define DI
def DI(X_in):
    rnd = np.random.randint(299, 330, size=1)[0]
    h_rem = 330 - rnd
    w_rem = 330 - rnd
    pad_top = np.random.randint(0, h_rem, size=1)[0]
    pad_bottom = h_rem - pad_top
    pad_left = np.random.randint(0, w_rem, size=1)[0]
    pad_right = w_rem - pad_left

    c = np.random.rand(1)
    if c <= 0.7:
        # bug
        # X_out = F.pad(F.interpolate(X_in, size=(rnd, rnd)), (pad_left, pad_top, pad_right, pad_bottom),
        # mode='constant', value=0)
        X_in_inter = F.interpolate(X_in, size=(rnd, rnd))
        X_out = F.pad(X_in_inter, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        return X_out
    else:
        return X_in


# define DI
def DI_keepresolution(X_in):
    rnd = np.random.randint(270, 299, size=1)[0]
    h_rem = 299 - rnd
    w_rem = 299 - rnd
    pad_top = np.random.randint(0, h_rem, size=1)[0]
    pad_bottom = h_rem - pad_top
    pad_left = np.random.randint(0, w_rem, size=1)[0]
    pad_right = w_rem - pad_left

    c = np.random.rand(1)
    if c <= 0.7:
        X_in_inter = F.interpolate(X_in, size=(rnd, rnd))
        X_out = F.pad(X_in_inter, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        return X_out
    else:
        return X_in


# define TI
def gkern(kernlen=15, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel
