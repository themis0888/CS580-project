import os
import math
import time
import datetime
from multiprocessing import Process
from multiprocessing import Queue

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import imageio

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torch.nn.functional as F

import numpy as np 
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import seaborn as sns
from scipy import stats
import pdb


# output, target should be torch tensor in [0, 255]
def calc_psnr(output, target, rgb_range = 255):
    if target.nelement() == 1: return 0
    
    shave = 6
    diff = (output - target) / rgb_range
    if diff.size(1) > 1:
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        diff = diff.mul(convert).sum(dim=1)

    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()
    
    return -10 * math.log10(mse)


def visualize_img(img):
    return np.clip(img, 0, 1)**0.454545 * 255
    