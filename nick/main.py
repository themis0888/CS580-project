from option import args
from trainer import Trainer
from tensorboardX import SummaryWriter
from data import KPCNDataset
from torch.utils.data import DataLoader

def main():
    writer = SummaryWriter()
    dataset = KPCNDataset()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_threads)
    kpcn_t = Trainer(args, loader, writer=writer)
    kpcn_t.train()

if __name__ == '__main__':
    main()