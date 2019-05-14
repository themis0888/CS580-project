from option import args
from trainer import Trainer
from tensorboardX import SummaryWriter
from data import KPCNDataset
from torch.utils.data import DataLoader
from download_patches import donwload_patches

def main():
    writer = SummaryWriter()
    # donwload_patches(args.remote_address, args.port, args.username, args.remote_path, args.dir_patch, args.n_patches)
    dataset = KPCNDataset()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_threads)
    kpcn_t = Trainer(args, loader, writer=writer)
    kpcn_t.train(epochs=args.epochs, learning_rate=args.lr)

if __name__ == '__main__':
    main()