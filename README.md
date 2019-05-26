# CS580-project

Project Repository for CS580.


## Setup
Make sure the right locals are set
```
$ export LC_ALL="en_US.UTF-8"
$ export LC_CTYPE="en_US.UTF-8"
```
Install necessary dependencies:
```
$ sudo apt-get install libopenexr-dev zlib1g-dev
```
Create a virtual environemnt
```
$ python3 -m venv env
```
Activate it
```
$ source env/bin/activate
```
First make sure `pip` is up-to-date by upgrading it
```
$ pip3 install --upgrade pip
```
Then install all requierements requiered for this project
```
$ pip3 install -r requierements.txt
```
Followed by pytorch
```
$ pip3 install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp35-cp35m-linux_x86_64.whl
$ pip3 install torchvision
```

## Run
ssh into machine
Make sure to activate environment
### Creates patches
```
python get_patches.py --download_images
```
### Train on patches
```
python main.py --batch_size 64
```
### With tensorboard
make sure to forward tensorboard port
```
-L 16006:127.0.0.1:6006
```
Tensorboard will be avaiable at:
```
http://127.0.0.1:16006
```
Start it with
```
```
