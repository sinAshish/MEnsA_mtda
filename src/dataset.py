#############################
# Author: Ashish Sinha
#############################

# import libraries
import os
import glob
import random
import h5py

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils import (normalize,
                   rotation_point_cloud,
                   jitter_point_cloud,
                   )

def load_dir(data_dir, name='train_files.txt'):
    """
    Reads the txt file and returns the path for samples
    """
    with open(os.path.join(data_dir,name),'r') as f:
        lines = f.readlines()
    return [os.path.join(data_dir, line.rstrip().split('/')[-1]) for line in lines]

class PointCloudDataset(Dataset):
    def __init__(
        self,
        root_dir,
        mode = 'train',
        num_points = 1024,
        aug = True,
        data_type = 'npy'
        ):
        super(PointCloudDataset, self).__init__()
        
        """
        Base Class for Modelnet and Shapent Datasets  
        Process the files/folders for easy reading of data.
        Input: 
            root_dir (str): path to dataset
            mode (str): train or test
            num_points (int): number of points in a sample
            aug (bool): data augmentation or not
        """
        
        self.root_dir = root_dir
        self.mode = mode
        self.num_points = num_points
        self.aug = aug
        self.data_type = data_type
        
        self.data = []
        self.label = []
        
        category = glob.glob(os.path.join(root_dir, '*'))
        category = [c.split(os.path.sep)[-1] for c in category]
        category = sorted(category)
        
        if mode == 'train':
            npy_list = glob.glob(os.path.join(root_dir, '*', 'train',f'*.{self.data_type}'))#[:50]
        else:
            npy_list = glob.glob(os.path.join(root_dir, '*', 'test',f'*.{self.data_type}'))#[50:80]
            
        for _dir in npy_list:
            self.data.append(_dir)
            self.label.append(category.index(_dir.split('/')[-3]))
        
class Modelnet10(PointCloudDataset):
    def __init__(
        self,
        root_dir,
        mode = 'train',
        num_points = 1024,
        aug = True,
        data_type = 'npy'
        ):
        super(Modelnet10, self).__init__(root_dir=root_dir, mode=mode)
        """
        Dataset class for Modelnet
        Input: 
            root_dir (str): path to dataset
            mode (str): train or test
            num_points (int): number of points in a sample
            aug (bool): data augmentation or not
        Ouput:
            point cloud: B x 3 x N x 1
            label: B x C
        """
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        #import pdb; pdb.set_trace()
        label = self.label[idx]
        pc = np.load(self.data[idx])[:self.num_points].astype(np.float32)
        pc = normalize(pc)
        
        if self.aug:
            pc = rotation_point_cloud(pc)
            pc = jitter_point_cloud(pc)
            
        pc = np.expand_dims(pc.transpose(), axis = 2)
        pc = torch.from_numpy(pc).type(torch.FloatTensor)
        
        return pc, label
    
class Shapenet(PointCloudDataset):
    def __init__(
        self,
        root_dir,
        mode = 'train',
        num_points = 1024,
        aug = True,
        data_type = 'npy'
        ):
        super(Shapenet, self).__init__(root_dir)
        """
        Dataset class for Shapenet
        Input: 
            root_dir (str): path to dataset
            mode (str): train or test
            num_points (int): number of points in a sample
            aug (bool): data augmentation or not
        Ouput:
            point cloud: B x 3 x N x 1
            label: B x C
        """
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #import pdb; pdb.set_trace()
        label = self.label[idx]
        if self.data_type == 'pts':
            pc = np.array([[float(value) for value in xyz.split(' ')]
                           for xyz in open(self.data[idx], 'r') if len(xyz.split(' ')) == 3])[:self.num_points, :]
        elif self.data_type == 'npy':
            pc = np.load(self.data[idx])[:self.num_points].astype(np.float32)
        pc = normalize(pc)
        if self.aug:
            pc = rotation_point_cloud(pc)
            pc = jitter_point_cloud(pc)
        pad = np.zeros(shape=(self.num_points - pc.shape[0], 3), dtype= float)
        pc = np.concatenate((pc, pad), axis=0)
        pc = np.expand_dims(pc.transpose(), axis =2)
        pc = torch.from_numpy(pc).type(torch.FloatTensor)
        
        return pc, label
    
class Scannet(Dataset):
    def __init__(
        self,
        root_dir,
        mode = 'train',
        aug= True,
        num_points=1024
    ):
        super(Scannet, self).__init__()
        
        """
        Dataset class for Scannet
        Input: 
            root_dir (str): path to dataset
            mode (str): train or test
            num_points (int): number of points in a sample
            aug (bool): data augmentation or not
        Ouput:
            point cloud: B x 3 x N x 1
            label: B x C
        """
        
        self.aug = aug
        self.num_points = num_points
        self.mode = mode
        
        point_list = list()
        label_list = list()
        
        if self.mode == 'train':
            data_path = load_dir(root_dir, name='train_files.txt')
        else:
            data_path = load_dir(root_dir, name = 'test_files.txt')
        for path in data_path:
            file = h5py.File(path, 'r')
            points = file['data'][:]
            label = file['label'][:]

            point_list.append(points)
            label_list.append(label)

        self.data = np.concatenate(point_list, axis=0)#[:50]
        self.labels = np.concatenate(label_list, axis=0)#[:50]
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        point_idx = np.arange(0, self.num_points)
        np.random.shuffle(point_idx)
        pc = self.data[idx][point_idx][:, :3]
        label = self.labels[idx]
        
        pc = normalize(pc)
        if self.aug:
            pc = rotation_point_cloud(pc)
            pc = jitter_point_cloud(pc)
        pc = np.expand_dims(pc.transpose(), axis = 2)
        pc = torch.from_numpy(pc).type(torch.FloatTensor)
        
        return pc, label

############################################################
# Wrapper to return dataloaders
# For each source dataset, the function returns a train and 
# test dataloader for the source and target domains
############################################################
def get_dataloaders(root_dir,
                    source, 
                    targets, 
                    batch_size = 64,
                    num_workers = 1,
                    pin_memory =  False,
                    drop_last = False):
    
    target_trainloaders = [0] * len(targets)
    target_testloaders = [0] * len(targets)
 
    modelnet_train_dataset = Modelnet10(root_dir=root_dir+'/modelnet/',
                                mode='train',
                                num_points=1024,
                                aug=True)
    modelnet_test_dataset = Modelnet10(root_dir= root_dir+'modelnet/',
                                        mode='test',
                                        num_points= 1024,
                                        aug=False)
    shapenet_train_dataset = Shapenet(root_dir=root_dir+'/shapenet/',
                                    mode='train',
                                    aug=True)
    shapenet_test_dataset = Shapenet(root_dir=root_dir+'/shapenet/',
                                    mode='test',
                                    aug=False)
    scannet_train_dataset = Scannet(root_dir=root_dir+'/scannet/',
                                    mode='train',
                                    aug=True)
    scannet_test_dataset = Scannet(root_dir=root_dir+'/scannet/',
                                    mode='test',
                                    aug=False)
    if source == 'modelnet':
        source_trainloader = DataLoader(modelnet_train_dataset,
                                    batch_size=batch_size,
                                    num_workers= num_workers,
                                    pin_memory= pin_memory,
                                    drop_last= drop_last,
                                    shuffle= True
                                    )
        source_testloader = DataLoader(modelnet_test_dataset,
                                    batch_size=batch_size,
                                    num_workers= num_workers,
                                    pin_memory= pin_memory,
                                    drop_last= drop_last
                                    )
        target_trainloaders[0] = DataLoader(shapenet_train_dataset,
                                    batch_size=batch_size,
                                    num_workers= num_workers,
                                    pin_memory= pin_memory,
                                    drop_last=  drop_last,
                                    shuffle=True
                                    )
        target_trainloaders[1] = DataLoader(scannet_train_dataset,
                                    batch_size=batch_size,
                                    num_workers= num_workers,
                                    pin_memory= pin_memory,
                                    drop_last=  drop_last,
                                    shuffle=True
                                    )
        target_testloaders[0] = DataLoader(shapenet_test_dataset,
                                    batch_size=batch_size,
                                    num_workers= num_workers,
                                    pin_memory= pin_memory,
                                    drop_last= drop_last
                                    )
        target_testloaders[1] = DataLoader(scannet_test_dataset,
                                    batch_size=batch_size,
                                    num_workers= num_workers,
                                    pin_memory= pin_memory,
                                    drop_last= drop_last
                                    )
    elif source == 'scannet':
        source_trainloader = DataLoader(scannet_train_dataset,
                                    batch_size=batch_size,
                                    num_workers= num_workers,
                                    pin_memory= pin_memory,
                                    drop_last= drop_last,
                                    shuffle=True
                                    )
        source_testloader = DataLoader(scannet_test_dataset,
                                    batch_size=batch_size,
                                    num_workers= num_workers,
                                    pin_memory= pin_memory,
                                    drop_last= drop_last
                                    )
        target_trainloaders[0] = DataLoader(shapenet_train_dataset,
                                    batch_size=batch_size,
                                    num_workers= num_workers,
                                    pin_memory= pin_memory,
                                    drop_last= drop_last,
                                    shuffle=True)
        target_trainloaders[1] = DataLoader(modelnet_train_dataset,
                                    batch_size=batch_size,
                                    num_workers= num_workers,
                                    pin_memory= pin_memory,
                                    drop_last= drop_last,
                                    shuffle=True
                                    )
        target_testloaders[0] = DataLoader(shapenet_test_dataset,
                                    batch_size=batch_size,
                                    num_workers= num_workers,
                                    pin_memory= pin_memory,
                                    drop_last= drop_last
                                    )
        target_testloaders[1] = DataLoader(modelnet_test_dataset,
                                    batch_size=batch_size,
                                    num_workers= num_workers,
                                    pin_memory= pin_memory,
                                    drop_last= drop_last
                                    )
    elif source == 'shapenet':
        source_trainloader = DataLoader(shapenet_train_dataset,
                                    batch_size=batch_size,
                                    num_workers= num_workers,
                                    pin_memory= pin_memory,
                                    drop_last= drop_last,
                                    shuffle=True
                                    )
        source_testloader = DataLoader(shapenet_test_dataset,
                                    batch_size=batch_size,
                                    num_workers= num_workers,
                                    pin_memory= pin_memory,
                                    drop_last= drop_last
                                    )
        target_trainloaders[0] = DataLoader(modelnet_train_dataset,
                                    batch_size=batch_size,
                                    num_workers= num_workers,
                                    pin_memory= pin_memory,
                                    drop_last= drop_last,
                                    shuffle=True
                                    )
        target_trainloaders[1] = DataLoader(scannet_train_dataset,
                                    batch_size=batch_size,
                                    num_workers= num_workers,
                                    pin_memory= pin_memory,
                                    drop_last= drop_last,
                                    shuffle=True
                                    )
        target_testloaders[0] = DataLoader(modelnet_test_dataset,
                                    batch_size=batch_size,
                                    num_workers= num_workers,
                                    pin_memory= pin_memory,
                                    drop_last= drop_last
                                    )
        target_testloaders[1] = DataLoader(scannet_test_dataset,
                                    batch_size=batch_size,
                                    num_workers= num_workers,
                                    pin_memory= pin_memory,
                                    drop_last= drop_last
                                    )
    
    return source_trainloader,\
            source_testloader,\
            target_trainloaders,\
            target_testloaders
