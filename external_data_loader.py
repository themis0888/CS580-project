from paramiko.client import SSHClient
from paramiko.sftp_client import SFTPClient
from paramiko import AutoAddPolicy

import getpass

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

        # Scan for total examples
        for scene_name in self.scenes:
            file_amount = len(self.sftp_client.listdir(scene_name))
            print("In {}: {}".format(scene_name, file_amount))

    def get_batch(idx, spp=128):
        pass

    def get_total(spp=128):
        pass
        

if __name__ == "__main__":
    password = getpass.getpass("Password: ")
    external_data_loader = ExternalDataLoader(password)
