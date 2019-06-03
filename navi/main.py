from option import args
from trainer import Trainer
from tensorboardX import SummaryWriter
from data import KPCNDataset
from torch.utils.data import DataLoader
import os
import datetime as dt



def main():
    now = dt.datetime.now()
    date_time = now.strftime('%Y-%m-%d/%H-%M')
    args.date_time = date_time

    args.log_dir = os.path.join('exp', args.dir_save, args.model, args.date_time)
    if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
    writer = SummaryWriter(log_dir = args.log_dir)

    train_set = KPCNDataset()
    test_set = KPCNDataset(train=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    kpcn_t = Trainer(args, train_loader, test_loader, writer=writer)
    if not args.test_only:
        kpcn_t.train(epochs=args.epochs)
    kpcn_t.test()

if __name__ == '__main__':
    main()