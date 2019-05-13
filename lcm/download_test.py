import os
import paramiko
import time
import getpass

if __name__ == '__main__':
    num_patches = 100

    password = getpass.getpass("Password: ")
    prev_time = time.time()
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('125.138.77.26', port=8385, username="tlcm", password=password)
    print("ssh connected")

    sftp = ssh.open_sftp()
    local_path = 'patches/'
    remote_path = "/media/tlcm/main hard/patches/"

    cnt = 0
    sftp.get(remote_path+'list.txt', 'list.txt')
    patch_save_dir = 'patches/'
    if not os.path.isdir(patch_save_dir):
            os.mkdir(patch_save_dir)
    with open('list.txt', 'r') as f:
        for name in f:
            name = name.rstrip()
            cnt += 1
            # print(remote_path+name)
            # print(patch_save_dir+name)
            sftp.get(remote_path+name, patch_save_dir+name)
            if cnt == num_patches:
                break

    print("time:", time.time() - prev_time)