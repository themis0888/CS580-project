import os
import paramiko
import time
import pyexr
import getpass

password = getpass.getpass("Password: ")
prev_time = time.time()
ssh = paramiko.SSHClient()
ssh.load_system_host_keys()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('125.138.77.26', port=8385, username="tlcm", password=password)
print("ssh connected")

sftp = ssh.open_sftp()
local_path = 'test.exr'
remote_path = "/media/tlcm/main hard/Dropbox/Tlcm/KAIST/19 spring/Computer Graphics/workspace/CS580-project/lcm/samples/"
# remote_path = "/media/tlcm/main hard/Dropbox/TLCM/KAIST/renderings/bathroom2/83069353-08192spp.exr"
for name in ["imp/10095699.npy", "proc/10095699.pickle"]:
    sftp.get(remote_path+name, name[name.index('/')+1:])

print("time:", time.time() - prev_time)

# f = pyexr.open(local_path)
