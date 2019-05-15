import torch
from network import Net
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from data import send_to_device
from data import to_torch_tensors
from data import show_data

import os
import time
from tqdm import tqdm
import pdb

class Trainer(): 
    def __init__(self, args, loader, writer = None):
        self.args = args
        self.model = args.model
        self.loader = loader
        self.device = args.device
        self.recon_kernel_size = self.args.recon_kernel_size
        self.eps = 0.00316
        self.global_step = 0
        self.model_dir = os.path.join('model', self.args.model)
        self.print_freq = self.args.print_freq
        self.writer = writer

        if not os.path.exists(self.model_dir): os.makedirs(self.model_dir)

    def train(self, epochs=200, learning_rate=1e-4, show_images=False):
        
        dataloader = self.loader

        # instantiate networks
        print('Making the Network')

        self.diffuseNet = Net(self.args).to(self.device)
        self.specularNet = Net(self.args).to(self.device)

        if self.args.debug:
            print(self.diffuseNet, "CUDA:", next(self.diffuseNet.parameters()).is_cuda)
            print(self.specularNet, "CUDA:", next(self.specularNet.parameters()).is_cuda)
        
        criterion = nn.L1Loss()

        optimizerDiff = optim.Adam(self.diffuseNet.parameters(), lr=learning_rate)
        optimizerSpec = optim.Adam(self.specularNet.parameters(), lr=learning_rate)

        if self.args.resume:
            self.diffuseNet.load_state_dict(torch.load(os.path.join(self.model_dir, 'diffuseNet.pt')))
            self.specularNet.load_state_dict(torch.load(os.path.join(self.model_dir, 'specularNet.pt')))

        accuLossDiff = 0
        accuLossSpec = 0
        accuLossFinal = 0
        writer_LossDiff = 0
        writer_LossSpec = 0
        
        # lDiff = []
        # lSpec = []
        # lFinal = []

        print('Start Training')
        start = time.time()
        permutation = [0, 3, 1, 2]

        # loader_start = time.time()
        for epoch in range(epochs):
            print('Epoch {:04d}'.format(epoch))
            for i_batch, sample_batched in enumerate(tqdm(dataloader, ncols = 80)):
                self.global_step += 1
                
                # loader_end = time.time()
                
                # get the inputs
                X_diff = sample_batched['X_diff'].permute(permutation).to(self.device)
                Y_diff = sample_batched['Reference'][:,:,:,:3].permute(permutation).to(self.device)

                # zero the parameter gradients
                optimizerDiff.zero_grad()

                # forward + backward + optimize
                outputDiff = self.diffuseNet(X_diff)

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
                outputSpec = self.specularNet(X_spec)

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

                    # if False:#i_batch % 500:
                    #     print("Sample, denoised, gt")
                    #     sz = 3
                    #     orig = self.crop_like(sample_batched['finalInput'].permute(permutation), outputFinal)
                    #     orig = orig.cpu().permute([0, 2, 3, 1]).numpy()[0,:]
                    #     img = outputFinal.cpu().permute([0, 2, 3, 1]).numpy()[0,:]
                    #     gt = self.crop_like(sample_batched['finalGt'].permute(permutation), outputFinal)
                    #     gt = gt.cpu().permute([0, 2, 3, 1]).numpy()[0,:]

                    Y_final = sample_batched['finalGt'].permute(permutation).to(self.device)
                    Y_final = self.crop_like(Y_final, outputFinal)
                    lossFinal = criterion(outputFinal, Y_final)
                    accuLossFinal += lossFinal.item()

                iter_lossDiff, iter_lossSpec = lossDiff.item(), lossSpec.item()
                accuLossDiff += iter_lossDiff
                accuLossSpec += iter_lossSpec
                writer_LossDiff += iter_lossDiff
                writer_LossSpec += iter_lossSpec
                # loader_start = time.time()

                if self.global_step-1 % self.print_freq ==0:
                    self.writer.add_scalars('data/LossDiff', {self.model: writer_LossDiff/self.print_freq}, self.global_step)
                    self.writer.add_scalars('data/LossSpec', {self.model: writer_LossSpec/self.print_freq}, self.global_step)
                    writer_LossDiff, writer_LossSpec = 0, 0
                    
                if self.global_step % self.args.save_freq == 0:
                    torch.save(self.diffuseNet.state_dict(), os.path.join(self.model_dir, 'diffuseNet.pt'))
                    torch.save(self.specularNet.state_dict(), os.path.join(self.model_dir, 'specularNet.pt'))
            

            print("Epoch {}".format(epoch + 1))
            print("LossDiff: {}".format(accuLossDiff))
            print("LossSpec: {}".format(accuLossSpec))
            print("LossFinal: {}".format(accuLossFinal))



            # lDiff.append(accuLossDiff)
            # lSpec.append(accuLossSpec)
            # lFinal.append(accuLossFinal)
            
            accuLossDiff = 0
            accuLossSpec = 0
            accuLossFinal = 0

        print('Finished training in mode', self.model)
        print('Took', time.time() - start, 'seconds.')
        
        # return self.diffuseNet, self.specularNet, lDiff, lSpec, lFinal

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

    def unsqueeze_all(self, d):
        for k, v in d.items():
            d[k] = torch.unsqueeze(v, dim=0)
        return d

    def denoise(self, data, debug=False, img_num=None):
        permutation = [0, 3, 1, 2]
        eps = 0.00316
        diffuseNet = self.diffuseNet.net
        specularNet = self.specularNet.net
        with torch.no_grad():
            # pdb.set_trace()
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
            outputFinal = outputDiff * (albedo + eps) + torch.exp(outputSpec) - 1.0

            if img_num is not None:
                # print("Sample, denoised, gt")
                sz = 15
                orig = self.crop_like(data['finalInput'].permute(permutation), outputFinal)
                orig = orig.cpu().permute([0, 2, 3, 1]).numpy()[0,:]
                if not os.path.isdir('figure'):
                    os.makedir('figure')
                show_data(orig, 'figure/{}_original.png'.format(img_num), figsize=(sz,sz), normalize=True)
                img = outputFinal.cpu().permute([0, 2, 3, 1]).numpy()[0,:]
                show_data(img, 'figure/{}_denoised.png'.format(img_num), figsize=(sz,sz), normalize=True)
                gt = self.crop_like(data['finalGt'].permute(permutation), outputFinal)
                gt = gt.cpu().permute([0, 2, 3, 1]).numpy()[0,:]
                show_data(gt, 'figure/{}_gt.png'.format(img_num), figsize=(sz,sz), normalize=True)

            Y_final = data['finalGt'].permute(permutation).to(self.device)
            Y_final = self.crop_like(Y_final, outputFinal)
            
            lossFinal = criterion(outputFinal, Y_final).item()
            
            if debug:
                print("LossDiff:", lossDiff)
                print("LossSpec:", lossSpec)
                print("LossFinal:", lossFinal)