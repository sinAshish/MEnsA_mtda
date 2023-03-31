import pdb
import glob
import zipfile
import torch

import numpy as np
from torch.autograd import Variable
from torch import optim

#################################
# Point Cloud Data Augmentations
#################################

def normalize(pc):
    """
    normalize point cloud in range L
    :param pc: type list
    :return: type list
    """
    pc_mean = pc.mean(axis=0)
    pc = pc - pc_mean
    pc_L_max = np.max(np.sqrt(np.sum(abs(pc ** 2), axis=-1)))
    pc = pc/pc_L_max
    return pc

def rotation_point_cloud(pc):
    """
    Randomly rotate the point clouds to augment the dataset
    rotation is per shape based along up direction
    :param pc: B X N X 3 array, original batch of point clouds
    :return: BxNx3 array, rotated batch of point clouds
    """
    # rotated_data = np.zeros(pc.shape, dtype=np.float32)

    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    # rotation_matrix = np.array([[cosval, 0, sinval],
    #                             [0, 1, 0],
    #                             [-sinval, 0, cosval]])
    rotation_matrix = np.array([[1, 0, 0],
                                [0, cosval, -sinval],
                                [0, sinval, cosval]])
    # rotation_matrix = np.array([[cosval, -sinval, 0],
    #                             [sinval, cosval, 0],
    #                             [0, 0, 1]])
    rotated_data = np.dot(pc.reshape((-1, 3)), rotation_matrix)

    return rotated_data

def rotate_point_cloud_by_angle(pc, rotation_angle):
    """
    Randomly rotate the point clouds to augment the dataset
    rotation is per shape based along up direction
    :param pc: B X N X 3 array, original batch of point clouds
    :param rotation_angle: angle of rotation
    :return: BxNx3 array, rotated batch of point clouds
    """
    # rotated_data = np.zeros(pc.shape, dtype=np.float32)

    # rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    rotated_data = np.dot(pc.reshape((-1, 3)), rotation_matrix)

    return rotated_data

def jitter_point_cloud(pc, sigma=0.01, clip=0.05):
    """
    Randomly jitter points. jittering is per point.
    :param pc: B X N X 3 array, original batch of point clouds
    :param sigma:
    :param clip:
    :return:
    """
    jittered_data = np.clip(sigma * np.random.randn(*pc.shape), -1 * clip, clip)
    jittered_data += pc
    return jittered_data

def shift_point_cloud(pc, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, shifted batch of point clouds
    """
    N, C = pc.shape
    shifts = np.random.uniform(-shift_range, shift_range, 3)
    pc += shifts
    return pc

def random_scale_point_cloud(pc, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, scaled batch of point clouds
    """
    N, C = pc.shape
    scales = np.random.uniform(scale_low, scale_high, 1)
    pc *= scales
    return pc

def rotate_perturbation_point_cloud(pc, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, rotated batch of point clouds
    """
    # rotated_data = np.zeros(pc.shape, dtype=np.float32)
    angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    shape_pc = pc
    rotated_data = np.dot(shape_pc.reshape((-1, 3)), R)
    return rotated_data

###############################
# helper functions for training
###############################

def adjust_learning_rate(optimizer, new_lr):
    """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
        
def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
def save_run_code(save_dir):
    f = glob.glob('**/*.py', recursive=True)
    zf = zipfile.ZipFile(f'{save_dir}/code.zip', 'w')
    for file in f:
        zf.write(file)
    zf.close()
    
def make_variable(x):
    x = Variable(x, volatile= True)
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def get_optimizer(args, lr, model):
    if args.optim == 'sgd':
        return optim.SGD(model.parameters(),
                         lr=lr, 
                         momentum = args.momentum,
                         weight_decay=args.weight_decay
                         )
    elif args.optim == 'adam':
        return optim.Adam(model.parameters(),
                          lr= lr, 
                          betas = [0.9, 0.999],
                          eps=1e-7,
                          weight_decay=args.weight_decay,
                          )
    elif args.optim == 'adamw':
        return optim.AdamW(model.parameters(),
                           lr = lr,
                           betas=(0.9, 0.999),
                           eps=1e-7,
                           weight_decay=args.weight_decay
                           )
    