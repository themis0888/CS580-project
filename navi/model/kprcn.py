from model import common

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pdb


def make_model(args, parent=False):
    return KPRCN(args)

class KPRCN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(KPRCN, self).__init__()
        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.nc_feats
        kernel_size = args.kernel_size
        reduction = args.reduction
        self.nalu = args.nalu 
        act = nn.ReLU(True)
        
        # define head module
        modules_head = [conv(args.nc_input, args.nc_feats, args.kernel_size),
			nn.ReLU()
            ]

        # define body module
        modules_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks-2)
        ]
        modules_body.append(conv(n_feats, n_feats, kernel_size))

        self.nc_output = 3 if args.prediction == 'DP' else args.recon_kernel_size**2
        
        # define tail module
        modules_tail = [conv(args.nc_feats, self.nc_output, args.kernel_size)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)


    def forward(self, x):
        # x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        if self.nalu:
            conv_W = F.conv2d(res, self.W, padding=1)
            conv_M = F.conv2d(res, self.M, padding=1)
            res = self.sig(conv_M) * self.tan(conv_W)
        
        # res += x

        x = self.tail(res)
        
        # x = self.add_mean(x)

        return x 
    
    def forward_debug(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)

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

