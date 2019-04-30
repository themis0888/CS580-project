
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')

import pyexr
from network import Net

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

import random, time
from random import randint
import pdb




class Trainer(): 
    def __init__(self, args, loader, my_model, my_loss, ckp, writer = None):
        self.args = args
        self.ckp = ckp
        self.loader = loader
        self.model = my_model
        self.loss = my_loss
        self.device = self.args.device
        self.recon_kernel_size = self.args.recon_kernel_size
        self.eps = self.args.eps
        self.global_step = 0
        


    def train(self, epochs=5, learning_rate=1e-4, show_images=False):
        
        dataloader = self.loader

        # instantiate networks
        print('Making the Network')

        # diffuseNet = make_net(self.args, self.args.n_resblocks, self.model).to(self.device)
        # specularNet = make_net(self.args, self.args.n_resblocks, self.model).to(self.device)
        diffuseNet = Net(self.args)
        specularNet = Net(self.args)


        print(diffuseNet, "CUDA:", next(diffuseNet.parameters()).is_cuda)
        print(specularNet, "CUDA:", next(specularNet.parameters()).is_cuda)
        
        criterion = nn.L1Loss()

        optimizerDiff = optim.Adam(diffuseNet.parameters(), lr=learning_rate)
        optimizerSpec = optim.Adam(specularNet.parameters(), lr=learning_rate)

        accuLossDiff = 0
        accuLossSpec = 0
        accuLossFinal = 0
        
        lDiff = []
        lSpec = []
        lFinal = []

        print('Start Training')
        start = time.time()
        permutation = [0, 3, 1, 2]
        # pdb.set_trace()
        loader_start = time.time()
        for epoch in range(epochs):
            for i_batch, sample_batched in enumerate(dataloader):
                loader_end = time.time()
                print(loader_end - loader_start)
                # pdb.set_trace()
                #print(i_batch)

                # get the inputs
                X_diff = sample_batched['X_diff'].permute(permutation).to(self.device)
                Y_diff = sample_batched['Reference'][:,:,:,:3].permute(permutation).to(self.device)

                # zero the parameter gradients
                optimizerDiff.zero_grad()

                # forward + backward + optimize
                outputDiff = diffuseNet(X_diff)

                # print(outputDiff.shape)

                if self.model == 'KPCN':
                    X_input = self.crop_like(X_diff, outputDiff)
                    outputDiff = self.apply_kernel(outputDiff, X_input)

                Y_diff = self.crop_like(Y_diff, outputDiff)

                lossDiff = criterion(outputDiff, Y_diff)
                lossDiff.backward()
                optimizerDiff.step()

                # get the inputs
                X_spec = sample_batched['X_spec'].permute(permutation).to(self.device)
                Y_spec = sample_batched['Reference'][:,:,:,3:6].permute(permutation).to(self.device)

                # zero the parameter gradients
                optimizerSpec.zero_grad()

                # forward + backward + optimize
                outputSpec = specularNet(X_spec)

                if self.model == 'KPCN':
                    X_input = self.crop_like(X_spec, outputSpec)
                    outputSpec = self.apply_kernel(outputSpec, X_input)

                Y_spec = self.crop_like(Y_spec, outputSpec)

                lossSpec = criterion(outputSpec, Y_spec)
                lossSpec.backward()
                optimizerSpec.step()

                # calculate final ground truth error
                with torch.no_grad():
                    albedo = sample_batched['origAlbedo'].permute(permutation).to(self.device)
                    albedo = self.crop_like(albedo, outputDiff)
                    outputFinal = outputDiff * (albedo + self.eps) + torch.exp(outputSpec) - 1.0

                    if False:#i_batch % 500:
                        print("Sample, denoised, gt")
                        sz = 3
                        orig = self.crop_like(sample_batched['finalInput'].permute(permutation), outputFinal)
                        orig = orig.cpu().permute([0, 2, 3, 1]).numpy()[0,:]
                        # show_data(orig, figsize=(sz,sz), normalize=True)
                        img = outputFinal.cpu().permute([0, 2, 3, 1]).numpy()[0,:]
                        # show_data(img, figsize=(sz,sz), normalize=True)
                        gt = self.crop_like(sample_batched['finalGt'].permute(permutation), outputFinal)
                        gt = gt.cpu().permute([0, 2, 3, 1]).numpy()[0,:]
                        # show_data(gt, figsize=(sz,sz), normalize=True)

                    Y_final = sample_batched['finalGt'].permute(permutation).to(self.device)

                    Y_final = self.crop_like(Y_final, outputFinal)

                    lossFinal = criterion(outputFinal, Y_final)

                    accuLossFinal += lossFinal.item()

                accuLossDiff += lossDiff.item()
                accuLossSpec += lossSpec.item()
                loader_start = time.time()
            

            print("Epoch {}".format(epoch + 1))
            print("LossDiff: {}".format(accuLossDiff))
            print("LossSpec: {}".format(accuLossSpec))
            print("LossFinal: {}".format(accuLossFinal))

            lDiff.append(accuLossDiff)
            lSpec.append(accuLossSpec)
            lFinal.append(accuLossFinal)
            
            accuLossDiff = 0
            accuLossSpec = 0
            accuLossFinal = 0

        print('Finished training in mode', self.model)
        print('Took', time.time() - start, 'seconds.')
        
        return diffuseNet, specularNet, lDiff, lSpec, lFinal



    def denoise(self, diffuseNet, specularNet, data, debug=False):
        with torch.no_grad():
            out_channels = diffuseNet[len(diffuseNet)-1].out_channels
            mode = 'DPCN' if out_channels == 3 else 'KPCN'
            criterion = nn.L1Loss()
            
            if debug:
                print("Out channels:", out_channels)
                print("Detected mode", mode)
            
            # make singleton batch
            data = send_to_device(to_torch_tensors(data))
            if len(data['X_diff'].size()) != 4:
                data = self.unsqueeze_all(data)
            
            print(data['X_diff'].size())
            
            X_diff = data['X_diff'].permute(permutation).to(self.device)
            Y_diff = data['Reference'][:,:,:,:3].permute(permutation).to(self.device)

            # forward + backward + optimize
            outputDiff = diffuseNet(X_diff)

            # print(outputDiff.shape)

            if mode == 'KPCN':
                X_input = self.crop_like(X_diff, outputDiff)
                outputDiff = self.apply_kernel(outputDiff, X_input)

            Y_diff = self.crop_like(Y_diff, outputDiff)

            lossDiff = criterion(outputDiff, Y_diff).item()

            # get the inputs
            X_spec = data['X_spec'].permute(permutation).to(self.device)
            Y_spec = data['Reference'][:,:,:,3:6].permute(permutation).to(self.device)

            # forward + backward + optimize
            outputSpec = specularNet(X_spec)

            if mode == 'KPCN':
                X_input = self.crop_like(X_spec, outputSpec)
                outputSpec = self.apply_kernel(outputSpec, X_input)

            Y_spec = self.crop_like(Y_spec, outputSpec)

            lossSpec = criterion(outputSpec, Y_spec).item()

            # calculate final ground truth error
            albedo = data['origAlbedo'].permute(permutation).to(self.device)
            albedo = self.crop_like(albedo, outputDiff)
            outputFinal = outputDiff * (albedo + self.eps) + torch.exp(outputSpec) - 1.0

            if True:
                print("Sample, denoised, gt")
                sz = 15
                orig = self.crop_like(data['finalInput'].permute(permutation), outputFinal)
                orig = orig.cpu().permute([0, 2, 3, 1]).numpy()[0,:]
                show_data(orig, figsize=(sz,sz), normalize=True)
                img = outputFinal.cpu().permute([0, 2, 3, 1]).numpy()[0,:]
                show_data(img, figsize=(sz,sz), normalize=True)
                gt = self.crop_like(data['finalGt'].permute(permutation), outputFinal)
                gt = gt.cpu().permute([0, 2, 3, 1]).numpy()[0,:]
                show_data(gt, figsize=(sz,sz), normalize=True)

            Y_final = data['finalGt'].permute(permutation).to(self.device)

            Y_final = self.crop_like(Y_final, outputFinal)
            
            lossFinal = criterion(outputFinal, Y_final).item()
            
            if debug:
                print("LossDiff:", lossDiff)
                print("LossSpec:", lossSpec)
                print("LossFinal:", lossFinal)

    def crop_like(self, data, like, debug=False):
        if data.shape[-2:] != like.shape[-2:]:
            # crop
            with torch.no_grad():
                dx, dy = data.shape[-2] - like.shape[-2], data.shape[-1] - like.shape[-1]
                data = data[:,:,dx//2:-dx//2,dy//2:-dy//2]
                if debug:
                    print(dx, dy)
                    print("After crop:", data.shape)
        return data

    def unsqueeze_all(self, d):
        for k, v in d.items():
            d[k] = torch.unsqueeze(v, dim=0)
        return d

    def apply_kernel(self, weights, data):
        # apply softmax to kernel weights
        weights = weights.permute((0, 2, 3, 1)).to(self.device)
        _, _, h, w = data.size()
        weights = F.softmax(weights, dim=3).view(-1, w * h, self.recon_kernel_size, self.recon_kernel_size)

        # now we have to apply kernels to every pixel
        # first pad the input
        r = self.recon_kernel_size // 2
        data = F.pad(data[:,:3,:,:], (r,) * 4, "reflect")
        
        #print(data[0,:,:,:])
        
        # make slices
        R = []
        G = []
        B = []
        kernels = []
        for i in range(h):
            for j in range(w):
                pos = i*h+j
                ws = weights[:,pos:pos+1,:,:]
                kernels += [ws, ws, ws]
                sy, ey = i+r-r, i+r+r+1
                sx, ex = j+r-r, j+r+r+1
                R.append(data[:,0:1,sy:ey,sx:ex])
                G.append(data[:,1:2,sy:ey,sx:ex])
                B.append(data[:,2:3,sy:ey,sx:ex])
                #slices.append(data[:,:,sy:ey,sx:ex])
                
        reds = (torch.cat(R, dim=1).to(self.device)*weights).sum(2).sum(2)
        greens = (torch.cat(G, dim=1).to(self.device)*weights).sum(2).sum(2)
        blues = (torch.cat(B, dim=1).to(self.device)*weights).sum(2).sum(2)

        
        res = torch.cat((reds, greens, blues), dim=1).view(-1, 3, h, w).to(self.device)
        
        return res
