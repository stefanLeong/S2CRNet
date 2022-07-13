# This is a sample Python script.
from __future__ import print_function, absolute_import

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import torch
import argparse
from scripts.utils.osutils import *
from scripts.utils.misc import save_checkpoint, adjust_learning_rate
import scripts.datasets as datasets
import scripts.machine as machines
from options import Options
import numpy as np
import random

######### Set Seeds ###########
random.seed(2333)
np.random.seed(2333)
torch.manual_seed(2333)
torch.cuda.manual_seed_all(2333)

def main(args):
    DataLoader = datasets.HCOCO

    train_loader = torch.utils.data.DataLoader(DataLoader('train', args), batch_size=args.train_batch, shuffle=True,
                                               num_workers=args.workers, pin_memory=False, drop_last=True)

    val_loader = torch.utils.data.DataLoader(DataLoader('val', args), batch_size=args.test_batch, shuffle=False,
                                             num_workers=args.workers, pin_memory=False)

    lr = args.lr

    data_loaders = (train_loader, val_loader)

    Machine = machines.__dict__[args.machine](datasets=data_loaders, args=args)

    for epoch in range(Machine.args.start_epoch, Machine.args.epochs):

        lr = adjust_learning_rate(data_loaders, Machine.optimizers, epoch, lr, args)
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))
        Machine.record('lr', lr, epoch)
        Machine.train(epoch)
        Machine.validate(epoch)
        save_checkpoint(Machine)


if __name__ == '__main__':
    parser = Options().init(argparse.ArgumentParser(description='PyTorch Training'))
    main(parser.parse_args())