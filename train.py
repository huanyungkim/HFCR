#train的使用方法： python trainor.py --save_path origin， 这里需要指定保存.pth文件的路径, 完整的保存路径为  ./weights/origin

import random
import numpy as np
import torch
from datasetflow import *
from netnew.newff import DepthFlow
from trainer.trainer2 import Trainer
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
    trainer = Trainer(model, optimizer, train_loader, val_set, save_name='duts_davis', save_step=100, val_step=20)#


    #原来使用的max_epoch为4000
    trainer.train(4000, path=options.save_path)   #


if __name__ == '__main__':

    # set device
    torch.cuda.set_device(0)
    #print("用FrequencyFuse替换掉了Decoder中的上采样")


    #print("当前CUDA设备索引:", torch.cuda.current_device())
    print("当前CUDA设备名称:", torch.cuda.get_device_name(torch.cuda.current_device()))
    print("num_workers = 12")
    print("lr = 1e-5, BatchSize=8")
    print(f'weights path: {options.save_path}')
    print(f'flow only')


    # define model
    ver = 'convnextv2'
    model = DepthFlow(ver).eval()  #这个.eval()不用管，实际上由Trainer控制

    # training stage
    model = torch.nn.DataParallel(model)

    #train_ytvos(model)
    train_duts_davis(model, ver)
