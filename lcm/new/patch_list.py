import os

path = "/media/tlcm/mainhard/patches/64/"
ls = os.listdir(path)
with open(path + 'list.txt', 'w') as f:
    for name in ls:
        f.write(name+'\n')