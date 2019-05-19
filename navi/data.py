from option import args

import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import torch
import torchvision
from scipy import ndimage
import pyexr
from PIL import Image

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
        self.train = train
        if train:
            self.names = glob.glob(os.path.join(args.dir_train,"*"))
            print("Train files:", len(self.names))
        else:
            tmp = glob.glob(os.path.join(args.dir_test,"*"))[0]
            self.ext = tmp[tmp.rindex('.')+1:]
            if self.ext == 'exr':
                self.names = glob.glob(os.path.join(args.dir_test, "*-00128spp.exr"))
                print("Test pairs:", len(self.names))
            elif self.ext == 'pt':
                self.names = glob.glob(os.path.join(args.dir_test, "*"))
                print("Test pairs:", len(self.names))
            else:
                raise TypeError('Not exr or pt file')
        

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if self.train:
            patch = torch.load(self.names[idx])
            return send_to_device(to_torch_tensors(patch))
        else:
            if self.ext == 'exr':
                img_num = self.names[idx][len(args.dir_test):-13]
                path128 = os.path.join(args.dir_test, img_num+'-00128spp.exr')
                path8192 = os.path.join(args.dir_test, img_num+'-08192spp.exr')
                pt_file = preprocess_input(path128, path8192)
                if args.save_test_proc:
                    if not os.path.exists(args.dir_test_proc):
                        os.makedirs(args.dir_test_proc)
                    torch.save(pt_file, os.path.join(args.dir_test_proc, img_num+'.pt'))
                return [img_num, pt_file]
            elif self.ext == 'pt':
                img_num = self.names[idx][len(args.dir_test):-3]
                return [img_num, torch.load(self.names[idx])]
            else:
                raise NotImplementedError('Sorry, I am late to update :(')

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

#########################
## show data, save img ##
#########################

def show_data(data, figure_path, figsize=(15, 15), normalize=False):
    if normalize:
        data = np.clip(data, 0, 1)**0.45454545
    fig = plt.figure(figsize=figsize)
    imgplot = plt.imshow(data, aspect='equal')
    imgplot.axes.get_xaxis().set_visible(False)
    imgplot.axes.get_yaxis().set_visible(False)
    fig.savefig(figure_path)
    plt.close(fig)

def save_img(data, img_path, normalize=True):
    if normalize:
        data = np.clip(data, 0, 1)**0.45454545
    if data.dtype == 'float32':
        data = (data * 255 / np.max(data)).astype('uint8')
    img = Image.fromarray(data)
    img.save(img_path)

#######################################
## pre-process and post-process data ##
#######################################
eps = 0.00316
def build_data(img):
    data = img.get()


def preprocess_diffuse(diffuse, albedo):
    return diffuse / (albedo + eps)


def postprocess_diffuse(diffuse, albedo):
    return diffuse * (albedo + eps)


def preprocess_specular(specular):
    assert(np.sum(specular < 0) == 0)
    return np.log(specular + 1)


def postprocess_specular(specular):
    return np.exp(specular) - 1


def preprocess_diff_var(variance, albedo):
    return variance / (albedo + eps)**2


def preprocess_spec_var(variance, specular):
    return variance / (specular+1e-5)**2


def gradients(data):
    h, w, c = data.shape
    dX = data[:, 1:, :] - data[:, :w - 1, :]
    dY = data[1:, :, :] - data[:h - 1, :, :]
    # padding with zeros
    dX = np.concatenate((np.zeros([h,1,c], dtype=np.float32),dX), axis=1)
    dY = np.concatenate((np.zeros([1,w,c], dtype=np.float32),dY), axis=0)
    
    return np.concatenate((dX, dY), axis=2)

def remove_channels(data, channels):
    for c in channels:
        if c in data:
            del data[c]
        else:
            print("Channel {} not found in data!".format(c))

            
# returns network input data from noisy .exr file
def preprocess_input(filename, gt):
    file = pyexr.open(filename)
    data = file.get_all()

    
    # just in case
    for k, v in data.items():
        data[k] = np.nan_to_num(v)
        
    file_gt = pyexr.open(gt)
    gt_data = file_gt.get_all()
    
    # just in case
    for k, v in gt_data.items():
        gt_data[k] = np.nan_to_num(v)
        
        
    # clip specular data so we don't have negative values in logarithm
    data['specular'] = np.clip(data['specular'], 0, np.max(data['specular']))
    data['specularVariance'] = np.clip(data['specularVariance'], 0, np.max(data['specularVariance']))
    gt_data['specular'] = np.clip(data['specular'], 0, np.max(data['specular']))
    gt_data['specularVariance'] = np.clip(gt_data['specularVariance'], 0, np.max(gt_data['specularVariance']))
        
        
    # save albedo
    data['origAlbedo'] = data['albedo'].copy()
        
    # save reference data (diffuse and specular)
    diff_ref = preprocess_diffuse(gt_data['diffuse'], gt_data['albedo'])
    spec_ref = preprocess_specular(gt_data['specular'])
    diff_sample = preprocess_diffuse(data['diffuse'], data['albedo'])
    
    data['Reference'] = np.concatenate((diff_ref[:,:,:3].copy(), spec_ref[:,:,:3].copy()), axis=2)
    data['Sample'] = np.concatenate((diff_sample, data['specular']), axis=2)
    
    # save final input and reference for error calculation
    # apply albedo and add specular component to get final color
    data['finalGt'] = gt_data['default']#postprocess_diffuse(data['Reference'][:,:,:3], data['albedo']) + data['Reference'][:,:,3:]
    data['finalInput'] = data['default']#postprocess_diffuse(data['diffuse'][:,:,:3], data['albedo']) + data['specular'][:,:,3:]
        
        
        
        
    # preprocess diffuse
    data['diffuse'] = preprocess_diffuse(data['diffuse'], data['albedo'])

    # preprocess diffuse variance
    data['diffuseVariance'] = preprocess_diff_var(data['diffuseVariance'], data['albedo'])

    # preprocess specular
    data['specular'] = preprocess_specular(data['specular'])

    # preprocess specular variance
    data['specularVariance'] = preprocess_spec_var(data['specularVariance'], data['specular'])

    # just in case
    data['depth'] = np.clip(data['depth'], 0, np.max(data['depth']))

    # normalize depth
    max_depth = np.max(data['depth'])
    if (max_depth != 0):
        data['depth'] /= max_depth
        # also have to transform the variance
        data['depthVariance'] /= max_depth * max_depth

    # Calculate gradients of features (not including variances)
    data['gradNormal'] = gradients(data['normal'][:, :, :3].copy())
    data['gradDepth'] = gradients(data['depth'][:, :, :1].copy())
    data['gradAlbedo'] = gradients(data['albedo'][:, :, :3].copy())
    data['gradSpecular'] = gradients(data['specular'][:, :, :3].copy())
    data['gradDiffuse'] = gradients(data['diffuse'][:, :, :3].copy())
    data['gradIrrad'] = gradients(data['default'][:, :, :3].copy())

    # append variances and gradients to data tensors
    data['diffuse'] = np.concatenate((data['diffuse'], data['diffuseVariance'], data['gradDiffuse']), axis=2)
    data['specular'] = np.concatenate((data['specular'], data['specularVariance'], data['gradSpecular']), axis=2)
    data['normal'] = np.concatenate((data['normalVariance'], data['gradNormal']), axis=2)
    data['depth'] = np.concatenate((data['depthVariance'], data['gradDepth']), axis=2)


    X_diff = np.concatenate((data['diffuse'],
                            data['normal'],
                            data['depth'],
                            data['gradAlbedo']), axis=2)

    X_spec = np.concatenate((data['specular'],
                            data['normal'],
                            data['depth'],
                            data['gradAlbedo']), axis=2)
    
    assert not np.isnan(X_diff).any()
    assert not np.isnan(X_spec).any()
    
    data['X_diff'] = X_diff
    data['X_spec'] = X_spec
    
    remove_channels(data, ('diffuseA', 'specularA', 'normalA', 'albedoA', 'depthA',
                                'visibilityA', 'colorA', 'gradNormal', 'gradDepth', 'gradAlbedo',
                                'gradSpecular', 'gradDiffuse', 'gradIrrad', 'albedo', 'diffuse', 
                                'depth', 'specular', 'diffuseVariance', 'specularVariance',
                                'depthVariance', 'visibilityVariance', 'colorVariance',
                                'normalVariance', 'visibility'))
    
    return data

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