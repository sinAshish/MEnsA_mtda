#############################################
# Author: Ashish Sinha
# Desc: Main wrapper to run the training code
#############################################

# import libraries
import os
import pdb
import sys
import json
import warnings
import argparse
import numpy as np
import json
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import (MultiStepLR, 
                                      StepLR, 
                                      ExponentialLR, 
                                      CosineAnnealingLR)
sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.utils import (seed_all,
                   save_run_code,
                   get_optimizer
                   )
from src.dataset import get_dataloaders
from models.model_mda import Net_MDA
from src.logger import get_logger
from src.trainer import train
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings('ignore')

# Command setting
parser = argparse.ArgumentParser(description='Main')
parser.add_argument('-source', '-s', type=str, help='source dataset', default='scannet')
parser.add_argument('-batchsize', '-b', type=int, help='batch size', default=64)
parser.add_argument('-gpu', '-g', type=str, help='cuda id', default='0')
parser.add_argument('-epochs', '-e', type=int, help='training epoch', default=100)
parser.add_argument('-curr_epoch', '-ce', type=int, help='starting epoch', default=1)
parser.add_argument('-num_workers', type=int, help='num workers', default=1)
parser.add_argument('-model', '-m', type=str, help='alignment model', default='MDA')
parser.add_argument('-lr',type=float, help='learning rate', default=0.001)
parser.add_argument('-alpha',type=float, help='alpha weight', default=0.2)
parser.add_argument('-weight_decay',type=float, help='decay weight', default=5e-4)
parser.add_argument('-init_beta',type=float, help='init beta weight', default=0.1)
parser.add_argument('-sch_kd_gamma',type=float, help='scheduler gamma weight', default=0.1)
parser.add_argument('-end_beta',type=float, help='end beta weight', default=0.9)
parser.add_argument('-momentum',type=float, help='gamma weight', default=0.9)
parser.add_argument('-scaler',type=float, help='scaler of learning rate', default=1.)
parser.add_argument('-log_interval',type=int, help='log interval', default=1000)
parser.add_argument('-save_interval',type=int, help='save checkpoint interval', default=25)
parser.add_argument('-resume','-r', default=None, help='resume training', type = str)
parser.add_argument('-datadir',type=str, help='directory of data', default='../data/PointDA_data/')
parser.add_argument('-log', type=str, help='directory of tb', default='saved/logs')
parser.add_argument('-save_dir', type=str, help='directory of tb', default='saved/ckpt')
parser.add_argument('-run_id', type=str, help= 'version for logging', default=1)
parser.add_argument('-optim', type=str, default='sgd', help='choose optimizer: adam/sgd/adamw')
parser.add_argument('-mixup', action='store_true', help='use mixup of features')
parser.add_argument('-mix_sep', action='store_true', help='allow features mixing of target sepearately')
parser.add_argument('-mixup_thres', type=float, default=0.3, help='threshold for using mixup')
parser.add_argument('-mix_type', type=int, default=-1, help='type of mixing')
parser.add_argument('-drop_last', action ='store_false')
parser.add_argument('-seed', default=42, type=int, help='seed value')

#loss weight parameters
parser.add_argument('-lambda_desc', type=float, default = -1., help='weight for descrepancy loss')
parser.add_argument('-lambda_mix', type=float, default = 1.2, help='weight for mixup loss')
parser.add_argument('-lambda_adv', type=float, default = 5., help='weight for adverserial loss')
parser.add_argument('-lambda_mmd', type=float, default = 5., help='weight for MMD loss')
parser.add_argument('-gamma',type=float, help='gamma weight for adverserial loss', default=0.5)

args = parser.parse_args()

############################
# Use datetime as a 'filename' for logging
############################

now = datetime.now()
now = now.strftime("%b-%m-%Y_%H:%M:%S")
args.run_id = now

LOG_DIR = args.log
dir_name = f"src_{args.source}_optim_{args.optim}_mixup_{args.mixup}_sep_{args.mix_sep}_Mixtype_{args.mix_type}_thres_{args.mixup_thres}_run_{args.run_id}"

############################
# Create Log and Checkpoint directories
############################

if not os.path.exists(os.path.join(os.getcwd(), LOG_DIR)):
    os.makedirs(os.path.join(os.getcwd(), LOG_DIR))
LOG_DIR = os.path.join(LOG_DIR, dir_name)
if not os.path.exists(os.path.join(os.getcwd(), LOG_DIR)):
    os.makedirs(os.path.join(os.getcwd(), LOG_DIR))

args.save_dir = os.path.join(args.save_dir, dir_name)
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

############################
# Save the hyperparameters in a file
############################
with open(os.path.join(args.save_dir, 'params.json'), 'w') as param:
    json.dump(vars(args), param)

save_run_code(args.save_dir)
writer = SummaryWriter(log_dir= LOG_DIR)

logger = get_logger(f'{LOG_DIR}/run_{args.run_id}')

if torch.cuda.is_available():
    device = 'cuda'
else:
    device='cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

device_ids = list(map(int, args.gpu.split(',')))
BATCH_SIZE = args.batchsize * len(args.gpu.split(',')) # increase the batch size if more than 1 gpu
LR = args.lr
weight_decay = args.weight_decay
momentum = args.momentum
epochs = args.epochs
root_dir = args.datadir
num_class = 10

logger.info(f'Given Hyperparameters: {vars(args)}')
logger.info(f'Training with {args.model} on gpu: {args.gpu} with batch size: {BATCH_SIZE} with {args.batchsize} on each.')
logger.info(f'Total Epochs: {epochs}')

# Set the source and target domains
if args.source == 'modelnet':
    targets=['shapenet', 'scannet']
elif args.source == 'shapenet':
    targets = ['modelnet', 'scannet']
elif args.source == 'scannet':
    targets = ['modelnet', 'shapenet']
    
logger.info(f'Multi Target Simultaneous Domain Adaptation from {args.source} --> {targets}')

########################################################
# Scheduling of a hyperparam for loss weights
#########################################################
growth_rate = torch.zeros(1)
if args.init_beta != 0.0:
    growth_rate = torch.log(torch.FloatTensor([args.end_beta/ args.init_beta]))/ torch.FloatTensor([epochs])

model = Net_MDA(model_name='mda').to(device)
optimizer = get_optimizer(args, LR, model)
scheduler = CosineAnnealingLR(optimizer, T_max= int(epochs*1.5))

if len(device_ids) > 1:
    model = nn.DataParallel(model).to(device)

global iteration
iteration = 0

save_name = f'best_{args.run_id}_{args.source}_vs_rest_da.pth'

########################################################
# Get source and target domain dataloaders
########################################################
source_trainloader, source_testloader, target_trainloaders, target_testloaders = get_dataloaders(root_dir,
                                                                                                 args.source,
                                                                                                 targets,
                                                                                                 BATCH_SIZE,
                                                                                                 args.num_workers,
                                                                                                 pin_memory= False,
                                                                                                 drop_last=args.drop_last)


############################
# Seed everything
############################
seed_all(args.seed)

#####################################################################################
# Run the training ccode
# Outputs: Best Mean sample accuracy and Best Mean per-class accuracy across domains
#####################################################################################
best_sample_acc, best_class_acc = train(args,
                                        writer,
                                        growth_rate,
                                        source_trainloader,
                                        target_trainloaders,
                                        target_testloaders,
                                        optimizer,
                                        scheduler,
                                        model,
                                        save_name,
                                        logger,
                                        device)

logger.info(f'Training finished with best sample acc of {best_sample_acc} % and best class sample acc of {best_class_acc} %')
