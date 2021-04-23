import os
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch.nn as nn
import argparse
from dataset.datasets import GL3D
from torch.utils.data import DataLoader
from utils import io_utils


from models.model import GNN
from loss import Loss
from train import trainer

import warnings


parser = argparse.ArgumentParser(description = "GCN with metric learning for view graph")
parser.add_argument('--train', type = bool, default= True)
parser.add_argument('--exp_name',type=str, default ='test')

#network
parser.add_argument('--emb_size', type = int, default= 768, help = 'Embedding size')
parser.add_argument('--emb_method', type = str, default='CNN')


#dataset
parser.add_argument('--dataset_root', type= str, default='/Volumes/Data2/GL3D/data', help='absolute dir of data')
parser.add_argument('--traindata_list', type = str, default='./data/train', help='graph data list in txt')
parser.add_argument('--testdata_list', type = str, default='./data/test')
parser.add_argument('--data_len', type = int, default= 100)

# training
parser.add_argument('--batch_size', type = int, default= 16)
parser.add_argument('--iterations', type = int, default= 20)
parser.add_argument('--lr', type = float, default= 0.0001)
parser.add_argument('--loss_function', type=str, default='dice', metavar='N')

parser.add_argument('--iteration', type= int, default=20)
parser.add_argument('--save_interval', type = int, default= 10)
parser.add_argument('--test_interval', type = int, default= 10)
parser.add_argument('--load_pretrained', type=bool, default=False)
parser.add_argument('--dec_lr', type=int, default=5, metavar='N',help='Decreasing the learning rate every x iterations')
parser.add_argument('--gamma', type = float, default=0.5, help='Learning rate Multiplicative factor  gamma')

args = parser.parse_args()

def init():

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    io = io_utils.IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    return io

def train(io):
    # datasets
    dataset = GL3D(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    args.batch_size = int(np.ceil(len(dataloader)/args.batch_size))

    # model
    gnn = GNN(args)
    io.cprint(str(gnn))

    # loss
    loss_func = Loss(args)

    trainer(args, dataloader, gnn, loss_func, io)

if __name__ == "__main__":
    io = init()
    if args.train:
        train(io)