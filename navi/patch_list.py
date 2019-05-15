import os

path = "/media/tlcm/main hard/patches/"
ls = os.listdir(path)
with open(path + 'list.txt', 'w') as f:
    for name in ls:
        f.write(name+'\n')