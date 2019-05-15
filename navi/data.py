from option1 import args

import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import torch
import torchvision
from scipy import ndimage

import random
from random import randint
import glob

import pdb
import time
import pickle
import os

figure_num = 0
eps = 0.00316

# set device
device = args.device
# device = torch.device('cpu')
# print(device)

############################
### Make a dataset class ###
############################

# cropped = to_torch_tensors(cropped)
# cropped = send_to_device(cropped)
# It contains None data for some index because of pruning data by importance sampling
class KPCNDataset(torch.utils.data.Dataset):
    def __init__(self, train=True):
        # get all name of data
        if train:
            patch_dir = args.dir_train
        else:
            patch_dir = args.dir_test
        self.names = glob.glob(os.path.join(patch_dir,"*"))
        print("Num files:", len(self.names))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        patch = torch.load(self.names[idx])        
        return send_to_device(to_torch_tensors(patch))

##########
## type ##
##########

def to_torch_tensors(data):
    if isinstance(data, dict):
        for k, v in data.items():
            if not isinstance(v, torch.Tensor):
                data[k] = torch.from_numpy(v)
    elif isinstance(data, list):
        for i, v in enumerate(data):
            if not isinstance(v, torch.Tensor):
                data[i] = to_torch_tensors(v)
        
    return data
 
def send_to_device(data):
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(device)
    elif isinstance(data, list):
        for i, v in enumerate(data):
            if isinstance(v, torch.Tensor):
                data[i] = v.to(device)        
    return data

###############
## show data ##
###############

def show_data(data, figure_path, figsize=(15, 15), normalize=False):
    if normalize:
        data = np.clip(data, 0, 1)**0.45454545
    fig = plt.figure(figsize=figsize)
    imgplot = plt.imshow(data, aspect='equal')
    imgplot.axes.get_xaxis().set_visible(False)
    imgplot.axes.get_yaxis().set_visible(False)
    fig.savefig(figure_path)
    plt.close(fig)

###############
## test code ##
###############

if __name__ == "__main__":
    dataset = KPCNDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4,
                                shuffle=True, num_workers=1)

    for i_batch, sample_batched in enumerate(dataloader):
        if i_batch == 2:
            break
        for i in range(4):
            fig = plt.figure()
            patch1 = sample_batched['finalInput'][i]
            patch2 = sample_batched['finalGt'][i]
            data1_ = np.clip(patch1, 0, 1)**0.45454545
            data2_ = np.clip(patch2, 0, 1)**0.45454545
            fig.add_subplot(1, 2, 1)
            imgplot = plt.imshow(data1_)
            imgplot.axes.get_xaxis().set_visible(False)
            imgplot.axes.get_yaxis().set_visible(False)
            fig.add_subplot(1, 2, 2)
            imgplot = plt.imshow(data2_)
            imgplot.axes.get_xaxis().set_visible(False)
            imgplot.axes.get_yaxis().set_visible(False)            
            fig.savefig('figure/{}_patch.png'.format(figure_num))
            figure_num += 1
            plt.close(fig)