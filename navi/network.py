import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import ipdb

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()

        self.args = args
        self.sig = nn.Sigmoid()
        self.tan = nn.Tanh()
        self.model = self.args.model
        self.n_layers = self.args.n_resblocks

        # define head module
        head = [
			nn.Conv2d(args.nc_input, args.nc_feats, args.kernel_size),
			nn.ReLU()
        ]
        
        body = []
        for l in range(self.n_layers-2):
            body += [
                    nn.Conv2d(args.nc_feats, args.nc_feats, args.kernel_size, padding=1),
                    nn.ReLU()
            ]
            
        self.nc_output = 3 if args.prediction == 'DP' else args.recon_kernel_size**2
        tail = [nn.Conv2d(args.nc_feats, self.nc_output, args.kernel_size)]#, padding=18)]
        
        for layer in head:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
        for layer in body:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
        for layer in tail:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)

        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)

    def forward(self, x):
        # ipdb.set_trace()
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)

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

