from paramiko.client import SSHClient
from paramiko.sftp_client import SFTPClient
from paramiko import AutoAddPolicy
from collections import OrderedDict

import getpass
import pyexr
import os
import math

class ExternalDataLoader:
    """
    Loads data from an external server
    """

    def __init__(self, password, scenes=["bathroom2", "car2", "classroom", "house", "room2", "room3", "spaceship", "staircase"], batch_size=64):
        self.scenes = scenes
        self.batch_size = batch_size
        
        self.client = SSHClient()
        self.client.load_system_host_keys()
        self.client.set_missing_host_key_policy(AutoAddPolicy())
        self.client.connect(hostname="143.248.38.66", port=3202, username="root", password=password)
        
        # For Documentation see here: http://docs.paramiko.org/en/2.4/api/sftp.html#paramiko.sftp_client.SFTPClient
        self.sftp_client = self.client.open_sftp()
        self.sftp_client.chdir("/home/siit/navi/data/input_data/deep_learning_denoising/renderings/")
        
        # print(self.sftp_client.listdir())
        self.possible_spp = [128, 256, 512, 1024, 8192] # 8192 is Ground Truth

        self.scene_files = OrderedDict()

        # Scan for total examples
        self.total_files = 0
        for scene_name in self.scenes:
            file_list = self.sftp_client.listdir(scene_name)
            file_amount = len(file_list)
            # print(self.sftp_client.listdir(scene_name))
            print("In {}: {}".format(scene_name, file_amount))
            self.total_files += file_amount
            self.scene_files[scene_name] = file_list

    
    @property
    def batches_amount(self):
        return math.ceil(self.files_amount / self.batch_size)
    
    @property
    def files_amount(self):
        return self.total_files / len(self.possible_spp)
        
    def get_data(self, idx, spp=128):
        """
        This returns an image and the corresponding GT
        """
        # Calculate scene
        if spp not in self.possible_spp:
            raise FileNotFoundError("There is no file for this spp.")

        total = 0
        scene_name = None
        for scene, file_list in self.scene_files.items():
            offset = int(len(file_list) / len(self.possible_spp))
            if idx > total + offset:
                total += offset
                continue
            scene_name = scene
            break

        all_file_spp = list(filter(lambda x: str(spp) in x, self.scene_files[scene_name]))
        all_file_spp.sort()

        sub_idx = idx - total
        file_name = all_file_spp[sub_idx]
        
        file_path = os.path.join(scene_name, file_name)
        self.sftp_client.get(file_path, "image.exr")
        file = pyexr.open("image.exr")

        return file
        

    def get_batch(self, idx, spp=128):
        X = []
        Y = []
        for pos in range(self.batch_size):
            x = self.get_data(idx * self.batch_size + pos, spp=spp)
            y = self.get_data(idx * self.batch_size + pos, spp=8192)
            X.append(x)
            Y.append(y)
        return X, Y

if __name__ == "__main__":
    password = getpass.getpass("Password: ")
    external_data_loader = ExternalDataLoader(password)

    print("----- Downloading Example File -----")
    file = external_data_loader.get_data(0)
    print("Width:", file.width)
    print("Height:", file.height)
    print("Available channels:")
    file.describe_channels()
    print("Default channels:", file.channel_map['default'])
    print("--------------- Done ---------------")

    # Takes a long time
    print("---- Loading a Batch of Size 64 ----")
    batch = external_data_loader.get_batch(0)
    print("------ Loaded Unique Values --------")
