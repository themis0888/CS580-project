import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import pdb

class KPCN(nn.Module):
    def __init__(self, args):
        super(KPCN, self).__init__()

        self.args = args
        self.sig = nn.Sigmoid()
        self.tan = nn.Tanh()
        self.model = self.args.model
        self.n_layers = self.args.n_resblocks

        # define head module
        layers = [
			nn.Conv2d(args.nc_input, args.nc_feats, args.kernel_size),
			nn.ReLU()
        ]
        
        for l in range(self.n_layers-2):
            layers += [
                    nn.Conv2d(args.nc_feats, args.nc_feats, args.kernel_size),
                    nn.ReLU()
            ]
            
        self.nc_output = 3 if args.prediction == 'DP' else args.recon_kernel_size**2
        layers += [nn.Conv2d(args.nc_feats, self.nc_output, args.kernel_size)]#, padding=18)]
        
        for layer in layers:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)

        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

