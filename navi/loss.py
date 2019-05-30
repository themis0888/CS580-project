import torch
import torch.nn as nn
from torch.nn import functional as func

import numpy as np

l1_loss_ = nn.L1Loss()

def LoG(img):
	weight = [
		[0, 0, 1, 0, 0],
		[0, 1, 2, 1, 0],
		[1, 2, -16, 2, 1],
		[0, 1, 2, 1, 0],
		[0, 0, 1, 0, 0]
	]
	weight = np.array(weight)

	weight_np = np.zeros((1, 1, 5, 5))
	weight_np[0, 0, :, :] = weight
	weight_np = np.repeat(weight_np, img.shape[1], axis=1)
	weight_np = np.repeat(weight_np, img.shape[0], axis=0)

	weight = torch.from_numpy(weight_np).type(torch.FloatTensor).to('cuda:0')

	return nn.functional.conv2d(img, weight, padding=1)

def l1_gradient_loss_(output, target):
    return torch.sum(torch.pow(LoG(output) - LoG(target), 2)) / torch.sum(torch.pow(LoG(target), 2))

def combined_loss_(output, target):
    ls = l1_loss_(output, target)
    lg = l1_gradient_loss_(output, target)
    return 0.8 * ls + 0.2 * lg

## Return callabable functions
def l1_loss():
    return l1_loss_

def l1_gradient_loss():
    return l1_gradient_loss_

def combined_loss():
    return combined_loss_
