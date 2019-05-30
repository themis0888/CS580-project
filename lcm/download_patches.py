import os
import paramiko
import time
import getpass
from tqdm import tqdm

def donwload_patches(remote_address, port, username, remote_path, local_path='/app/data/', num_patches=10000000):
    password = getpass.getpass("Password: ")
    # prev_time = time.time()
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(remote_address, port=port, username=username, password=password)
    print("ssh connected")

    sftp = ssh.open_sftp()
    # local_path = 'patches/'
    # remote_path = "/media/tlcm/main hard/patches/"

    cnt = 0
    sftp.get(os.path.join(remote_path,'list.txt'), 'list.txt')
    if not os.path.isdir(local_path):
            os.mkdir(local_path)
    with open('list.txt', 'r') as f:
        for name in tqdm(f):
            name = name.rstrip()
            cnt += 1
            # print(remote_path+name)
            # print(patch_save_dir+name)
            sftp.get(os.path.join(remote_path+name), os.path.join(local_path+name))
            if cnt == num_patches:
                break

if __name__ == '__main__':
    remote_path = "/media/tlcm/mainhard/patches/64/"
    prev_time = time.time()
    donwload_patches('125.138.77.26', port=8385, username='tlcm', remote_path=remote_path)
    print('Time:', time.time() - prev_time)