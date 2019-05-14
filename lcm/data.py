from option import args

import numpy as np
import pyexr
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
patch_size = args.patch_size # patches are 64x64
eps = 0.00316
patch_dir = args.dir_patch

# set device
device = args.device
# device = torch.device('cpu')
# print(device)

# delete preprocessed and importance map file after running
# for time checking and other reasons
# file_delete = False

############################
### Make a dataset class ###
############################

# cropped = to_torch_tensors(cropped)
# cropped = send_to_device(cropped)
# It contains None data for some index because of pruning data by importance sampling
class KPCNDataset(torch.utils.data.Dataset):
    def __init__(self):
        # get all name of data
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
## test code ##
###############

if __name__ == "__main__":
    dataset = KPCNDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4,
                                shuffle=True, num_workers=1)
    # prev_time = time.time()
    for i_batch, sample_batched in enumerate(dataloader):
        # pdb.set_trace()
        if i_batch == 2:
            break
            

        # show each patches
        
        # pdb.set_trace()
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
    
    # print(time.time() - prev_time)

    # clear file
    # for file in os.listdir("samples/imp"):
    #     os.remove("samples/imp/"+file)
    # os.rmdir("samples/imp")
    # for file in os.listdir("samples/proc"):
    #     os.remove("samples/proc/"+file)
    # os.rmdir("samples/proc")
