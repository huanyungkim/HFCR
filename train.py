import random
import numpy as np
import torch
from data import *
from hfcr import HFCR
from trainer_hfcr import Trainer
import warnings

from optparse import OptionParser
warnings.filterwarnings('ignore')
from utils import set_seed

set_seed(3407)

parser = OptionParser()

parser.add_option('--save_path', action='store', dest='save_path', default=None, help='Save path for .pth files')

options, args = parser.parse_args()



def train_duts_davis(model, ver):

    duts_set = TrainDUTSv2('DB/DUTSv2', clip_n=384)
    davis_set = TrainDAVIS('DB/DAVIS', '2016', 'train', clip_n=128)

    train_set = torch.utils.data.ConcatDataset([duts_set, davis_set])


    ''' data_loader'''
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True, num_workers=12, pin_memory=True)#
    val_set = TestDAVIS('DB/DAVIS', '2016', 'val')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    trainer = Trainer(model, optimizer, train_loader, val_set, save_name='duts_davis', save_step=500, val_step=100)#

    trainer.train(4000, path=options.save_path)   




if __name__ == '__main__':

    # set device
    torch.cuda.set_device(0)
    # define model
    ver = 'convnextv2'
    model = HFCR(ver).eval()
    # training stage
    model = torch.nn.DataParallel(model)
    train_duts_davis(model, ver)
