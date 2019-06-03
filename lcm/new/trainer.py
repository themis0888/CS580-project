import torch
from network import Net
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from data import send_to_device, to_torch_tensors, show_data

import os
import time
from tqdm import tqdm
import pdb
import utils
import pytorch_ssim

class Trainer(): 
    def __init__(self, args, train_loader, test_loader, writer = None):
        self.args = args
        self.model = args.model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = args.device
        self.recon_kernel_size = self.args.recon_kernel_size
        self.eps = 0.00316
        self.global_step = 0
        self.model_dir = os.path.join('model', self.args.model)
        self.print_freq = self.args.print_freq
        self.writer = writer

        if not os.path.exists(self.model_dir): os.makedirs(self.model_dir)
        
        print('Making the Network')
        self.diffuseNetList = [Net(self.args).to(self.device)]*5
        # self.diffuseNet = Net(self.args).to(self.device)
        self.specularNet = Net(self.args).to(self.device)

    def train(self, epochs=200, learning_rate=1e-4, show_images=False):
        
        dataloader = self.train_loader

        # if self.args.debug:
        #     print(self.diffuseNet, "CUDA:", next(self.diffuseNet.parameters()).is_cuda)
        #     print(self.specularNet, "CUDA:", next(self.specularNet.parameters()).is_cuda)
        
        criterion = nn.L1Loss()
        # criterion = pytorch_ssim.SSIM()

        optimizerDiffList = []
        for i in range(len(self.diffuseNetList)):
            optimizerDiffList.append(optim.Adam(self.diffuseNetList[i].parameters(), lr=learning_rate))
        # optimizerDiff = optim.Adam(self.diffuseNet.parameters(), lr=learning_rate)
        optimizerSpec = optim.Adam(self.specularNet.parameters(), lr=learning_rate)
        schedulerDiffList = []
        for i in range(len(optimizerDiffList)):
            schedulerDiffList.append(torch.optim.lr_scheduler.MultiStepLR(schedulerDiffList[i], [10**4], gamma=0.01))
        schedulerSpec = torch.optim.lr_scheduler.MultiStepLR(optimizerSpec, [10**4], gamma=0.01)
        # schedulerDiff = torch.optim.lr_scheduler.StepLR(optimizerDiff, step_size=10**4, gamma=0.1)
        # schedulerSpec = torch.optim.lr_scheduler.StepLR(optimizerSpec, step_size=10**4, gamma=0.1)

        if self.args.resume:
            for i in range(len(self.diffuseNetList)):
                self.diffuseNetList[i].load_state_dict(torch.load(os.path.join(self.model_dir, 'diffuseNet{}.pt'.format(i))))
            self.specularNet.load_state_dict(torch.load(os.path.join(self.model_dir, 'specularNet.pt')))

        writer_LossDiff = 0
        writer_LossSpec = 0
        writer_LossFinal = 0
        writer_SSIMDiff = 0
        writer_SSIMSpec = 0
        writer_SSIMFinal = 0
        tr_fin_best = 100

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
                # if torch.isinf(outputDiff).any():
                #     print("Found Infinity Values in Kernel, Going to Skip this Batch!")
                #     continue
                # if torch.isnan(outputDiff).any():
                #     print("Found NaN Values in Kernel, Going to Skip this Batch!")
                #     continue

                # print(outputDiff.shape)

                if self.model == 'KPCN':
                    X_input = self.crop_like(X_diff, outputDiff)
                    outputDiff = self.apply_kernel(outputDiff, X_input)
                
                # if torch.isinf(outputDiff).any():
                #     print("Found Infinity Values in Output, Going to Skip this Batch!")
                #     continue
                # if torch.isnan(outputDiff).any():
                #     print("Found NaN Values in Output, Going to Skip this Batch!")
                #     continue

                Y_diff = self.crop_like(Y_diff, outputDiff)

                # lossDiff = criterion(outputDiff * 255, Y_diff * 255)
                # lossDiff = -criterion(outputDiff, Y_diff)
                lossDiff = criterion(outputDiff, Y_diff)
                ssimDiff = pytorch_ssim.ssim(outputDiff, Y_diff)
                lossDiff.backward()
                 # torch.nn.utils.clip_grad_norm_(self.diffuseNet.parameters(), 5)
                # torch.nn.utils.clip_grad_value_(self.diffuseNet.parameters(), 10)

                # Check if gradient became NaN --> Skip            
                # found_NaN = False
                # for param in self.diffuseNet.parameters():
                #     if not torch.isnan(param.grad.data).any():
                #         continue
                #     found_NaN = True
                #     break
                # if found_NaN:
                #     print("Found NaN Values in Gradient, Goint to Skip this Batch!")
                #     continue

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

                # lossSpec = criterion(outputSpec * 255, Y_spec * 255)
                # lossSpec = -criterion(outputSpec, Y_spec)
                lossSpec = criterion(outputSpec, Y_spec)
                ssimSpec = pytorch_ssim.ssim(outputSpec, Y_spec)
                lossSpec.backward()
                optimizerSpec.step()

                schedulerDiff.step()
                schedulerSpec.step()

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
                    ssimFinal = pytorch_ssim.ssim(outputFinal, Y_final)
                    # accuLossFinal += lossFinal.item()

                iter_lossDiff, iter_lossSpec, iter_lossFinal = lossDiff.item(), lossSpec.item(), lossFinal.item()
                # accuLossDiff += iter_lossDiff
                # accuLossSpec += iter_lossSpec
                writer_LossDiff += iter_lossDiff
                writer_LossSpec += iter_lossSpec
                writer_LossFinal += iter_lossFinal
                writer_SSIMDiff += 1-ssimDiff.item()
                writer_SSIMSpec += 1-ssimSpec.item()
                writer_SSIMFinal += 1-ssimFinal.item()
                # loader_start = time.time()

                if (self.global_step-1) % self.print_freq == 0:
                    self.writer.add_scalars('data/L1Loss', {'LossDiff': writer_LossDiff/self.print_freq,
                                                        'LossSpec':writer_LossSpec/self.print_freq,
                                                        'LossFinal':writer_LossFinal/self.print_freq}, self.global_step)
                    self.writer.add_scalars('data/SSIM', {'SSIMDiff': writer_SSIMDiff/self.print_freq,
                                                        'SSIMSpec':writer_SSIMSpec/self.print_freq,
                                                        'SSIMFinal':writer_SSIMFinal/self.print_freq}, self.global_step)
                    # self.writer.add_scalars('data/LossSpec', {self.model: writer_LossSpec/self.print_freq}, self.global_step)
                    # self.writer.add_scalars('data/LossFinal', {self.model: writer_LossFinal/self.print_freq}, self.global_step)
                    # self.writer.add_scalars('data/SSIMDiff', {self.model: writer_SSIMDiff/self.print_freq}, self.global_step)
                    # self.writer.add_scalars('data/SSIMSpec', {self.model: writer_SSIMSpec/self.print_freq}, self.global_step)
                    # self.writer.add_scalars('data/SSIMFinal', {self.model: writer_SSIMFinal/self.print_freq}, self.global_step)
                    writer_LossDiff, writer_LossSpec, writer_LossFinal = 0, 0, 0
                    writer_SSIMDiff, writer_SSIMSpec, writer_SSIMFinal = 0, 0, 0

                    if tr_fin_best > writer_LossFinal/self.print_freq:
                        tr_fin_best = writer_LossFinal/self.print_freq
                        torch.save(self.diffuseNet.state_dict(), os.path.join(self.model_dir, 'fin_best_diff.pt'))
                        torch.save(self.specularNet.state_dict(), os.path.join(self.model_dir, 'fin_best_spec.pt'))
                        
                if self.global_step % self.args.save_freq == 0:
                    torch.save(self.diffuseNet.state_dict(), os.path.join(self.model_dir, 'diffuseNet.pt'))
                    torch.save(self.specularNet.state_dict(), os.path.join(self.model_dir, 'specularNet.pt'))
                    
                if self.global_step % self.args.test_freq == 0:
                    self.test()
                
                
            

            # print("Epoch {}".format(epoch + 1))
            # print("LossDiff: {}".format(accuLossDiff))
            # print("LossSpec: {}".format(accuLossSpec))
            # print("LossFinal: {}".format(accuLossFinal))



            # lDiff.append(accuLossDiff)
            # lSpec.append(accuLossSpec)
            # lFinal.append(accuLossFinal)
            
            # accuLossDiff = 0
            # accuLossSpec = 0
            # accuLossFinal = 0

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
        if self.args.only_test:
            self.diffuseNet.load_state_dict(torch.load(os.path.join(self.model_dir, 'diffuseNet.pt')))
            self.specularNet.load_state_dict(torch.load(os.path.join(self.model_dir, 'specularNet.pt')))
        for i_batch, tp in enumerate(tqdm(self.test_loader)):
            img_num, test_data = tp
            self.denoise(test_data, img_num[0])

    def denoise(self, data, img_num, debug=False):
        permutation = [0, 3, 1, 2]
        eps = 0.00316
        diffuseNet = self.diffuseNet
        specularNet = self.specularNet
        with torch.no_grad():
            # pdb.set_trace()
            out_channels = diffuseNet.net[len(diffuseNet.net)-1].out_channels
            mode = 'DPCN' if out_channels == 3 else 'KPCN'
            criterion = nn.L1Loss()
            # criterion = pytorch_ssim.SSIM()
            
            if debug:
                print("Out channels:", out_channels)
                print("Detected mode", mode)
            
            # make singleton batch
            data = send_to_device(to_torch_tensors(data), self.device)
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
            ssimDiff = 1-pytorch_ssim.ssim(outputDiff, Y_diff)

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
            ssimSpec = 1-pytorch_ssim.ssim(outputSpec, Y_spec)
            

            # calculate final ground truth error
            albedo = data['origAlbedo'].permute(permutation).to(self.device)
            albedo = self.crop_like(albedo, outputDiff)
            outputFinal = outputDiff * (albedo + eps) + torch.exp(outputSpec) - 1.0

            # save image
            img = outputFinal.cpu().permute([0, 2, 3, 1]).numpy()[0,:]         
            img = np.clip(img, 0, 1)**0.45454545
            new_img = (img * 255 / np.max(img)).astype('uint8')
            new_img1 = np.transpose(new_img, [2, 0, 1])

            Y_final = data['finalGt'].permute(permutation).to(self.device)
            Y_final = self.crop_like(Y_final, outputFinal)

            # save image
            img = Y_final.cpu().permute([0, 2, 3, 1]).numpy()[0,:]         
            img = np.clip(img, 0, 1)**0.45454545
            new_img = (img * 255 / np.max(img)).astype('uint8')
            new_img2 = np.transpose(new_img, [2, 0, 1])

            Y_orig = data['finalInput'].permute(permutation).to(self.device)
            Y_orig = self.crop_like(Y_orig, outputFinal)
            img = Y_orig.cpu().permute([0, 2, 3, 1]).numpy()[0,:]         
            img = np.clip(img, 0, 1)**0.45454545
            new_img = (img * 255 / np.max(img)).astype('uint8')
            new_img3 = np.transpose(new_img, [2, 0, 1])

            new_img = np.concatenate((new_img3, new_img1, new_img2), axis=2)
            self.writer.add_image('img/'+str(img_num), new_img, self.global_step)
            
            lossFinal = criterion(outputFinal, Y_final).item()
            ssimFinal = 1-pytorch_ssim.ssim(outputFinal, Y_final)
            # psnrFinal = utils.calc_psnr(outputFinal, Y_final)

            self.writer.add_scalars('data/l1', {'{}_lossDiff'.format(img_num): lossDiff, '{}_lossSpec'.format(img_num) : lossSpec, '{}_lossFinal'.format(img_num) : lossFinal}, self.global_step)
            self.writer.add_scalars('data/ssim', {'{}_ssimDiff'.format(img_num): ssimDiff, '{}_ssimSpec'.format(img_num):ssimSpec, '{}_ssimFinal'.format(img_num):ssimFinal}, self.global_step)