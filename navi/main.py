import torch
from torch.utils.data import DataLoader
import data
from option import args
from trainer import Trainer
from tensorboardX import SummaryWriter

torch.manual_seed(args.seed)
import pdb


def main():
    global model
    writer = SummaryWriter()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = data.KPCNDataset(args)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_threads)
    model = 'KPCN' # model.Model(args, checkpoint)
    loss1 = None # loss.Loss(args, checkpoint) if not args.test_only else None
    checkpoint = None 

    kpcn_t = Trainer(args, loader, model, loss1, checkpoint, writer = writer)
    # pdb.set_trace()

    while 1:
        # t.save_prob()
        
        kpcn_t.train()
        # t.test()
            
        # checkpoint.done()

if __name__ == '__main__':
    main()
