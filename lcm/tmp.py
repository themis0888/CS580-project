import glob
import os

names = glob.glob('/app/data/train/*_1.pt')
for i, name in enumerate(names):
    img_num = name[name.rindex('/')+1:name.rindex('_')]
    try:
        os.remove('/app/data/exr/{}-00128spp.exr'.format(img_num))
        os.remove('/app/data/exr/{}-08192spp.exr'.format(img_num))
    except FileNotFoundError:
        pass
