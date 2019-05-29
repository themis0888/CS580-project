import argparse
import torch
# import template

parser = argparse.ArgumentParser()

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--only_test', action='store_true',
                    help='No train, only test')
# parser.add_argument('--template', default='.',
#                     help='You can set various templates in option.py')
parser.add_argument('--clean_up', action='store_true',
                    help='Remove all previously runs and results')

# Hardware specifications
# parser.add_argument('--n_threads', type=int, default=0,
#                     help='number of threads for data loading')
# parser.add_argument('--cpu', action='store_true',
#                     help='use cpu only')
# parser.add_argument('--n_GPUs', type=int, default=1,
#                     help='number of GPUs')
# parser.add_argument('--seed', type=int, default=1,
#                     help='random seed')
parser.add_argument('--device', default='cuda',
                    help='using cpu or gpu')

# Data specifications
parser.add_argument('--dir_train', type=str, default='/home/ubuntu/data/train/',
                    help='train dataset directory')
parser.add_argument('--dir_test', type=str, default='/home/ubuntu/data/test/',
                    help='test dataset directory')
parser.add_argument('--save_test_proc', action='store_true',
                    help='Save processed test file')
parser.add_argument('--dir_test_proc', type=str, default='test_proc/',
                    help='processed test dataset directory')
# parser.add_argument('--patch_size', type=int, default=64,
#                     help='output patch size')
parser.add_argument('--model_name', type=str, default=None,
                    help='Model Name for Log Dir')

# Model specifications
parser.add_argument('--model', default='KPCN',
                    choices=('KPCN', 'DPCN'),
                    help='model name, KPCN(default) or DPCN')

parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
# parser.add_argument('--pre_train', type=str, default='',
#                     help='pre-trained model directory')
# parser.add_argument('--extend', type=str, default='.',
#                     help='pre-trained model directory')
parser.add_argument('--n_resblocks', type=int, default=9,
                    help='number of residual blocks')
parser.add_argument('--nc_feats', type=int, default=100,
                    help='number of channels in feature maps')
parser.add_argument('--nc_input', type=int, default=28,
                    help='number of channels of input')
parser.add_argument('--nc_output', type=int, default=3,
                    help='number of channels of output')
parser.add_argument('--kernel_size', type=int, default=5,
                    help='number of kernel')
parser.add_argument('--recon_kernel_size', type=int, default=21,
                    help='number of kernel')
# parser.add_argument('--res_scale', type=float, default=1,
#                     help='residual scaling')
# parser.add_argument('--shift_mean', default=True,
#                     help='subtract pixel mean from the input')
# parser.add_argument('--dilation', action='store_true',
#                     help='use dilated convolution')
# parser.add_argument('--precision', type=str, default='single',
#                     choices=('single', 'half'),
#                     help='FP precision for test (single | half)')
# parser.add_argument('--nalu', action='store_true',
#                     help='use dilated convolution')
parser.add_argument('--batch_norm', action='store_true',
                    help='use batch normalization')
parser.add_argument('--dropout', action='store_true',
                    help='use dropout')     


# Option for Residual dense network (RDN)
# parser.add_argument('--G0', type=int, default=64,
#                     help='default number of filters. (Use in RDN)')
# parser.add_argument('--RDNkSize', type=int, default=3,
#                     help='default kernel size. (Use in RDN)')
# parser.add_argument('--RDNconfig', type=str, default='B',
#                     help='parameters config of RDN. (Use in RDN)')

# Option for Residual channel attention network (RCAN)
# parser.add_argument('--n_resgroups', type=int, default=10,
#                     help='number of residual groups')
# parser.add_argument('--reduction', type=int, default=16,
#                     help='number of feature maps reduction')

# Training specifications
# parser.add_argument('--reset', action='store_true',
#                     help='reset the training')
parser.add_argument('--resume', action='store_true',
                    help='restart learning')
# parser.add_argument('--test_every', type=int, default=1000,
#                     help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=8,
                    help='input batch size for training')
# parser.add_argument('--split_batch', type=int, default=1,
#                     help='split the batch into smaller chunks')
# parser.add_argument('--self_ensemble', action='store_true',
#                     help='use self-ensemble method for test')
# parser.add_argument('--test_only', action='store_true',
#                     help='set this option to test the model')
# parser.add_argument('--gan_k', type=int, default=1,
#                     help='k value for adversarial loss')
parser.add_argument('--fixed_seed', type=int, default=-1,
                    help='Fixed seed for reproducibility')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
# parser.add_argument('--decay', type=str, default='20',
#                     help='learning rate decay type')
# parser.add_argument('--gamma', type=float, default=0.5,
#                     help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
# parser.add_argument('--weight_decay', type=float, default=0,
#                     help='weight decay')
# parser.add_argument('--gclip', type=float, default=0,
#                     help='gradient clipping threshold (0 = no clipping)')

# Loss specifications
# parser.add_argument('--loss', type=str, default='1*L1',
#                     help='loss function configuration')
# parser.add_argument('--skip_threshold', type=float, default='1e8',
#                     help='skipping batch that has large error')

# Log specifications
# parser.add_argument('--save', type=str, default='test',
#                     help='file name to save')
# parser.add_argument('--load', type=str, default='',
#                     help='file name to load')
# parser.add_argument('--save_models', action='store_true',
#                     help='save all intermediate models')
parser.add_argument('--save_freq', type=int, default=100,
                    help='save all intermediate models')
parser.add_argument('--test_freq', type=int, default=100,
                    help='save all intermediate models')
parser.add_argument('--print_freq', type=int, default=100,
                    help='how many batches to wait before logging training status')
# parser.add_argument('--save_results', action='store_true',
#                     help='save output results')

args = parser.parse_args()

def to_dir_path(path):
    path = path if path[-1] == '/' else path+'/'
    return path

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
    elif arg.startswith('dir_'):
        vars(args)[arg] = to_dir_path(vars(args)[arg])

args.device = args.device if torch.cuda.is_available() else 'cpu'
print("Device: ", args.device)
