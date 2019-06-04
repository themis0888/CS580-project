from datetime import datetime
import os
import math

def make_folder_with_time(path='.'):
    dic = {1:'Jan', 2:'Feb', 3:"Mar", 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
    now = datetime.now()
    dt = '{}{}_{}-{}'.format(dic[now.month], now.day, now.hour, now.minute)
    os.makedirs(os.path.join(path, dt))
    return dt

def calc_psnr(output, target, rgb_range = 255):
    if target.nelement() == 1: return 0
    
    shave = 6
    diff = (output - target) / rgb_range
    if diff.size(1) > 1:
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        diff = diff.mul(convert).sum(dim=1)

    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)