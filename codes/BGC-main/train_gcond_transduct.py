from deeprobust.graph.data import Dataset
import numpy as np
import random
import time
import argparse
import torch
from utils import *
import torch.nn.functional as F
from gcond_agent_transduct import GCond
from utils_graphsaint import DataGraphSAINT


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=3, help='gpu id')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--dis_metric', type=str, default='ours')
parser.add_argument('--epochs', type=int, default=5000)
#hyperparamters for gnn
parser.add_argument('--nlayers', type=int, default=3)
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--lr_adj', type=float, default=0.01)
parser.add_argument('--lr_feat', type=float, default=0.01)
parser.add_argument('--lr_model', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--keep_ratio', type=float, default=1.0)
parser.add_argument('--reduction_rate', type=float, default=1)
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--alpha', type=float, default=0, help='regularization term.')
parser.add_argument('--debug', type=int, default=0)
#hyperparameters for graph condensation
parser.add_argument('--sgc', type=int, default=1)
parser.add_argument('--inner', type=int, default=0)
parser.add_argument('--outer', type=int, default=20)
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--one_step', type=int, default=0)
#hyper parameters for backdoor
parser.add_argument('--backdoor', type=bool, default=True)
parser.add_argument('--trigger_size', type=int, default=3)
parser.add_argument('--attach_rate', type=float, default=0.15)
parser.add_argument('--target_class', type=int, default=0)
parser.add_argument('--threshold', type=float, default=0.1)
parser.add_argument('--lr_trigger', type=float, default=0.01)
parser.add_argument('--target_loss_weight', type=float, default=1.0)
parser.add_argument('--outer_epochs_backdoor', type=int, default=1)#400)
parser.add_argument('--attack_rate_test',type=float,default=0.5)
parser.add_argument('--last_multi', type=bool, default=False)
parser.add_argument('--dis_weight',type=float, default=1)
parser.add_argument('--selector',type=str, default='cluster') #'random'
parser.add_argument('--defense_type',type=str, default=None) #'prune', None, 'rand_smooth'
parser.add_argument('--prune_rate',type=float, default=0.2) #'prune', None, 'rand_smooth'

args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)

# random seed setting
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print(args)

data_graphsaint = ['flickr', 'reddit', 'ogbn-arxiv']
if args.dataset in data_graphsaint:
    data = DataGraphSAINT(args.dataset)
    data_full = data.data_full
else:
    data_full = get_dataset(args.dataset, args.normalize_features)
    data = Transd2Ind(data_full, keep_ratio=args.keep_ratio)

agent = GCond(data, args, device='cuda')

agent.train()
