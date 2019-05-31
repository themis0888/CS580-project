from option import args
from trainer import Trainer
from tensorboardX import SummaryWriter
from data import KPCNDataset
from torch.utils.data import DataLoader
import shutil
import os

import torch
import numpy as np

def main():
    if args.clean_up:
        if os.path.isdir("runs/"):
            shutil.rmtree("runs/")
        if os.path.isdir("result/"):
            shutil.rmtree("result/")

    if args.fixed_seed != -1:
        print("Using seed: {}".format(args.fixed_seed))
        torch.manual_seed(args.fixed_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.fixed_seed)

    writer = SummaryWriter(logdir="runs/"+args.model_name)
    train_set = KPCNDataset()
    test_set = KPCNDataset(train=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)
    kpcn_t = Trainer(args, train_loader, test_loader, writer=writer)
    if not args.only_test:
        kpcn_t.train(epochs=args.epochs)
    kpcn_t.test()

if __name__ == '__main__':
    main()
