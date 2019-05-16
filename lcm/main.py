from option1 import args
from trainer import Trainer
from tensorboardX import SummaryWriter
from data import KPCNDataset
from torch.utils.data import DataLoader
from download_patches import donwload_patches


def main():
    writer = SummaryWriter()
    train_set = KPCNDataset()
    test_set = KPCNDataset(train=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)
    kpcn_t = Trainer(args, train_loader, test_loader, writer=writer)
    if not args.only_test:
        kpcn_t.train(epochs=args.epochs, learning_rate=args.lr)
    kpcn_t.test()

if __name__ == '__main__':
    main()