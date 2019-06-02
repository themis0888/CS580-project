'''
CUDA_VISIBLE_DEVICES=2 python main.py --model KPCN --dir_save debug --test_freq 200 --print_freq 20 --visualize_freq 20
'''


import torch
from network import Net
import model.kprcan
import model.kprcn
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils

from data import send_to_device, to_torch_tensors, show_data, save_img

import os
import time
from tqdm import tqdm
import utility
import cv2
import numpy as np
import ipdb

class Trainer(): 
    def __init__(self, args, train_loader, test_loader, writer = None):
        self.args = args
        self.model = args.model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = args.device
        self.recon_kernel_size = self.args.recon_kernel_size
        self.eps = 0.00316
        self.global_step = 1 # args.global_step
        self.log_dir = os.path.join('exp', self.args.dir_save)
        self.model_dir = os.path.join(self.log_dir, 'model', self.args.model)
        self.print_freq = self.args.print_freq
        self.visualize_freq = self.args.visualize_freq
        self.writer = writer

        if not os.path.exists(self.model_dir): os.makedirs(self.model_dir)
        
        print('Making the Network')
        if args.model == 'KPRCAN':
            self.diffuseNet = model.kprcan.KPRCAN(self.args).to(self.device)
            self.specularNet = model.kprcan.KPRCAN(self.args).to(self.device)
        elif args.model == 'KPRCN':
            self.diffuseNet = model.kprcn.KPRCN(self.args).to(self.device)
            self.specularNet = model.kprcn.KPRCN(self.args).to(self.device)
        else:
            self.diffuseNet = Net(self.args).to(self.device)
            self.specularNet = Net(self.args).to(self.device)

    def train(self, epochs=200, show_images=False):

        learning_rate = self.args.lr
        dataloader = self.train_loader

        if self.args.debug:
            print(self.diffuseNet, "CUDA:", next(self.diffuseNet.parameters()).is_cuda)
            print(self.specularNet, "CUDA:", next(self.specularNet.parameters()).is_cuda)
        
        criterion = nn.L1Loss()

        optimizerDiff = optim.Adam(self.diffuseNet.parameters(), lr=learning_rate)
        optimizerSpec = optim.Adam(self.specularNet.parameters(), lr=learning_rate/10)
        # pdb.set_trace()
        if self.args.resume:
            self.diffuseNet.load_state_dict(torch.load(os.path.join(self.model_dir, 'diff_best.pt')))
            self.specularNet.load_state_dict(torch.load(os.path.join(self.model_dir, 'spec_best.pt')))
            print('Best model downloaded')

        accuLossDiff = 0
        accuLossSpec = 0
        accuLossFinal = 0
        writer_LossDiff = []
        writer_LossSpec = []
        writer_psnr = []
        
        # lDiff = []
        # lSpec = []
        # lFinal = []

        print('Start Training')
        start = time.time()
        permutation = [0, 3, 1, 2]
        tr_diff_best, tr_spec_best, tr_total_best, L_spec_log_best = 100, 10, 10, 1E-9
        # loader_start = time.time()
        phase = 1
        self.print_freq = self.args.print_freq // 4
        self.test_freq = self.args.print_freq
        for epoch in range(epochs):
            print('Epoch {:04d}'.format(epoch))
            for i_batch, sample_batched in enumerate(dataloader):

                if (self.global_step > self.args.print_freq * 10) and (phase == 1): 
                    phase = 2
                    self.print_freq = self.args.print_freq
                    self.test_freq = self.args.test_freq
                    optimizerDiff = optim.Adam(self.diffuseNet.parameters(), lr=learning_rate/5)
                    optimizerSpec = optim.Adam(self.specularNet.parameters(), lr=learning_rate/500)
                    print('Learning Rate Changed to {:.4E}'.format(learning_rate/5))
                
                self.global_step += 1
                loader_end = time.time()
                
                # get the inputs
                X_diff = sample_batched['X_diff'].permute(permutation).to(self.device)
                Y_diff = sample_batched['Reference'][:,:,:,:3].permute(permutation).to(self.device)

                # zero the parameter gradients
                optimizerDiff.zero_grad()

                # forward + backward + optimize
                outputDiff = self.diffuseNet(X_diff)

                # print(outputDiff.shape)

                if self.model != 'DPCN':
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

                if self.model != 'DPCN':
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

                    Y_final = sample_batched['finalGt'].permute(permutation).to(self.device)
                    Y_final = self.crop_like(Y_final, outputFinal)
                    lossFinal = criterion(outputFinal, Y_final)
                    accuLossFinal += lossFinal.item()

                out_data = torch.clamp(outputFinal, 0, 1)**0.454545
                gt_data = torch.clamp(Y_final, 0, 1)**0.454545
                psnr = utility.calc_psnr(out_data*255, gt_data*255)                    
                # pdb.set_trace()
                iter_lossDiff, iter_lossSpec = lossDiff.item(), lossSpec.item()
                accuLossDiff += iter_lossDiff
                accuLossSpec += iter_lossSpec
                writer_LossDiff.append(iter_lossDiff)
                writer_LossSpec.append(iter_lossSpec)
                writer_psnr.append(psnr)

                
                
                if (self.global_step) % self.visualize_freq == 0:
                    # in_data = torch.clamp(X_fin_input, 0, 1)**0.454545
 
                    print_step = self.global_step // self.visualize_freq * self.visualize_freq
                    X_input = self.crop_like(sample_batched['finalInput'].permute(permutation).to(self.device), outputDiff)
                    in_data = torch.clamp(vutils.make_grid(X_input, normalize = True, scale_each=True), 0, 1)**0.454545
                    out_data = torch.clamp(vutils.make_grid(outputFinal, normalize = True, scale_each=True), 0, 1)**0.454545
                    gt_data = torch.clamp(vutils.make_grid(Y_final, normalize = True, scale_each=True), 0, 1)**0.454545
                    cat = torch.cat((in_data.unsqueeze(0), out_data.unsqueeze(0), gt_data.unsqueeze(0)), dim=0)
                    total = vutils.make_grid(cat, normalize = True, scale_each=True, nrow = 1)
                    self.writer.add_image('data/Training/Total Summary', total, print_step)
                    # self.writer.add_image('data/Training/Input data', in_data, print_step)
                    # self.writer.add_image('data/Training/Out data', out_data, print_step)
                    # self.writer.add_image('data/Training/Target data', gt_data, print_step)


                if ((self.global_step) % self.print_freq == 0) and (self.global_step > 5):
                    
                    print_step = self.global_step // self.print_freq * self.print_freq
                    # ipdb.set_trace()
                    L_diff = sum(writer_LossDiff)/len(writer_LossDiff)
                    L_spec = sum(writer_LossSpec)/len(writer_LossSpec)
                    L_psnr = sum(writer_psnr)/len(writer_psnr)

                    self.writer.add_scalars('data/Train/LossDiff', {self.model: L_diff}, print_step)
                    self.writer.add_scalars('data/Train/LossSpec', {self.model: L_spec}, print_step)
                    self.writer.add_scalars('data/Train/PSNR', {self.model: L_psnr}, print_step)
                    print('[{:6d}step] L_diff: {:.3E} \t L_sepc: {:.3E} \t PSNR: {:.2f}'.format(print_step, L_diff, L_spec, L_psnr))
                    writer_LossDiff, writer_LossSpec, writer_psnr = [], [], []
                    
                    
                    if L_spec_log_best > L_spec:
                        learning_rate = learning_rate*0.2
                        optimizerSpec = optim.Adam(self.specularNet.parameters(), lr=learning_rate)
                        L_spec_log_best = L_spec_log_best/100
                        print('Learning rate changed to {:.2E}'.format(learning_rate))
                    
                    if tr_diff_best > L_diff:
                        tr_diff_best = L_diff
                        torch.save(self.diffuseNet.state_dict(), os.path.join(self.model_dir, 'diff_best.pt'))
                        print('Best diff_Net saved')
                    
                    if tr_spec_best > L_spec:
                        tr_spec_best = L_spec
                        torch.save(self.specularNet.state_dict(), os.path.join(self.model_dir, 'spec_best.pt'))
                        print('Best spec_Net saved')
                    
                    if tr_total_best > L_spec + L_diff:
                        tr_total_best = L_spec + L_diff
                        torch.save(self.specularNet.state_dict(), os.path.join(self.model_dir, 'total_spec_best.pt'))
                        torch.save(self.diffuseNet.state_dict(), os.path.join(self.model_dir, 'total_diff_best.pt'))
                        print('Best Net saved')

                    
                if self.global_step % self.args.save_freq == 0:
                    torch.save(self.diffuseNet.state_dict(), os.path.join(self.model_dir, 'diffuseNet.pt'))
                    torch.save(self.specularNet.state_dict(), os.path.join(self.model_dir, 'specularNet.pt'))
                    print('Latest Model saved')
                    

                if (self.global_step-1) % self.args.test_freq == 0:
                    self.test()
                loader_start = time.time()
            

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
    
    
    def test(self):
        if self.args.test_only:
            self.diffuseNet.load_state_dict(torch.load(os.path.join(self.model_dir, 'diff_best.pt')))
            self.specularNet.load_state_dict(torch.load(os.path.join(self.model_dir, 'spec_best.pt')))
        psnrs = []
        
        for i_batch, tp in enumerate(tqdm(self.test_loader)):
            # ipdb.set_trace()
            img_num, test_data = tp
            if ('10064185' not in img_num) and self.global_step < 1000: continue
            losses, result = self.denoise(test_data, img_num[0])
            psnrs.append(losses[-1])
            if '10064185' in img_num:
                in_data, out_data, gt_data = result
                cat = torch.cat((in_data.unsqueeze(0), out_data.unsqueeze(0), gt_data.unsqueeze(0)), dim=0)
                total = vutils.make_grid(cat, normalize = True, scale_each=True, nrow = 3)
        # self.writer.add_image('data/Test/Input', in_data, self.global_step)
        # self.writer.add_image('data/Test/Output', out_data, self.global_step)
        # self.writer.add_image('data/Test/Target', gt_data, self.global_step)
        self.writer.add_image('data/Test/Total Summary', total, self.global_step)
        avg_psnr = sum(psnrs)/len(psnrs)
        print('Avg. PSNR: {:.3f}'.format(avg_psnr))
        self.writer.add_scalars('data/Test/PSNR', {self.model: avg_psnr}, self.global_step)


    def denoise(self, data, img_num, debug=False):
        permutation = [0, 3, 1, 2]
        eps = 0.00316
        diffuseNet = self.diffuseNet

        specularNet = self.specularNet
        with torch.no_grad():
            # pdb.set_trace()
            out_channels = diffuseNet.nc_output
            mode = 'DPCN' if out_channels == 3 else 'KPCN'
            criterion = nn.L1Loss()
            
            if debug:
                print("Out channels:", out_channels)
                print("Detected mode", mode)
            
            # make singleton batch
            data = send_to_device(to_torch_tensors(data))
            if len(data['X_diff'].size()) != 4:
                data = self.unsqueeze_all(data)
            if debug:
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

            # Save result images
            img = outputFinal.cpu().permute([0, 2, 3, 1]).numpy()[0,:]
            if not os.path.exists('result'):
                os.makedirs('result')
            save_img(img, 'result/{}_denoised.png'.format(img_num))

            Y_final = data['finalGt'].permute(permutation).to(self.device)
            Y_final = self.crop_like(Y_final, outputFinal)

            X_fin_input = data['finalInput'].permute(permutation).to(self.device)
            X_fin_input = self.crop_like(X_fin_input, Y_final)


            lossFinal = criterion(outputFinal, Y_final).item()
            
            if debug:
                print("LossDiff:", lossDiff)
                print("LossSpec:", lossSpec)
                print("LossFinal:", lossFinal)
            
            # ipdb.set_trace()
            in_data = torch.clamp(X_fin_input, 0, 1)**0.454545
            out_data = torch.clamp(outputFinal, 0, 1)**0.454545
            gt_data = torch.clamp(Y_final, 0, 1)**0.454545
            psnr = utility.calc_psnr(out_data*255, gt_data*255)
            losses = [lossDiff, lossSpec, lossFinal, psnr]
            result = [in_data[0], out_data[0], gt_data[0]]

            return losses, result