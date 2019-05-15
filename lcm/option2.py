import argparse

parser = argparse.ArgumentParser(description='Get patches from exr images')

def to_dir_path(path):
    path = path if path[-1] == '/' else path+'/'
    return path

parser.add_argument('--dir_data', '-d', type=str, default='images/',
                    help='dataset directory')
parser.add_argument('--dir_patch', '-p', type=str, default='/home/ubuntu/data',
                    help='patchset directory')                                                            
parser.add_argument('--patch_size', '-s', type=int, default=64,
                    help='output patch size')
parser.add_argument('--n_patches', '-n', type=int, default=400,
                    help='the number of pathces')
# parser.add_argument('--debug', action='store_true',
#                     help='Enables debug mode')
parser.add_argument('--make_list', action='store_true',
                    help='Make list.txt which shows list of files')
parser.add_argument('--check_time', action='store_true',
                    help='Show elapsed time to get patch')
parser.add_argument('--download_images', action='store_true',
                    help='Downloads the images before hand')

args = parser.parse_args()

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
    elif arg.startswith('dir_'):
        vars(args)[arg] = to_dir_path(vars(args)[arg])
