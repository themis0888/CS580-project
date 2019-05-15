from option1 import args
from trainer import Trainer
from tensorboardX import SummaryWriter
from data import KPCNDataset
from torch.utils.data import DataLoader
from download_patches import donwload_patches
from get_patches import preprocess_input

def main():
    writer = SummaryWriter()
    dataset = KPCNDataset()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    kpcn_t = Trainer(args, loader, writer=writer)
    kpcn_t.train(epochs=args.epochs, learning_rate=args.lr)
    img_num = '10499343'
    test_data = preprocess_input('test/'+img_num+'-00128spp.exr', 'test/'+img_num+'-08192spp.exr')
    kpcn_t.denoise(test_data, debug=True, img_num=img_num)

if __name__ == '__main__':
    main()