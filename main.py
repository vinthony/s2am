from __future__ import print_function, absolute_import

import argparse
import torch

from scripts.utils.misc import save_checkpoint, adjust_learning_rate
import scripts.utils.pytorch_ssim as pytorch_ssim
import scripts.datasets as datasets
import scripts.machines as machines
from options import Options

def main(args):

    DataLoader = datasets.COCO

    if 'mmu' in args.arch:
        DataLoader = datasets.COCOv2

    if args.task == 'inpainting':
        DataLoader = datasets.Inpainting

    train_loader = torch.utils.data.DataLoader(DataLoader('train',args),batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=False)
    
    val_loader = torch.utils.data.DataLoader(DataLoader('val',args),batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=False)


    lr = args.lr

    data_loaders = (train_loader,val_loader)

    Machine = machines.__dict__[args.machine](datasets=data_loaders, args=args)


    for epoch in range(Machine.args.start_epoch, Machine.args.epochs):

        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))
        lr = adjust_learning_rate(data_loaders, Machine.optimizer, epoch, lr, args)

        Machine.record('lr',lr, epoch)        
        Machine.train(epoch)
        Machine.validate(epoch)
        save_checkpoint(Machine)

if __name__ == '__main__':
    parser=Options().init(argparse.ArgumentParser(description='PyTorch Training'))
    main(parser.parse_args())
