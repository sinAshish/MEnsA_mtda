import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks import (transform_net,
                      conv_2d,
                      GradReverse,
                      fc_layer,
                      Discriminator)

    
class Pointnet_cls(nn.Module):
    def __init__(self, num_class=10):
        super(Pointnet_cls, self).__init__()
        self.trans_net1 = transform_net(3,3)
        self.trans_net2 = transform_net(64,64)
        self.conv1 = conv_2d(3, 64, 1)
        self.conv2 = conv_2d(64, 64, 1)
        self.conv3 = conv_2d(64, 64, 1)
        self.conv4 = conv_2d(64, 128, 1)
        self.conv5 = conv_2d(128, 1024, 1)

        self.mlp1 = fc_layer(1024, 512)
        self.dropout1 = nn.Dropout2d(p=0.7)
        self.mlp2 = fc_layer(512, 256)
        self.dropout2 = nn.Dropout2d(p=0.7)
        self.class_cls = nn.Linear(256, num_class)

        self.domain_classifier = Discriminator(256, 4096)
        
    def get_adversarial_result(self, x, source = True, delta=1.):
        loss_fn = nn.BCELoss()
        loss_fn = loss_fn.to(x.device)
        if source:
            domain_label = torch.ones(x.size(0)).long().to(x.device)
        else:
            domain_label = torch.zeros(x.size(0)).long().to(x.device)
            
        x = GradReverse.apply(x, delta)
        domain_pred = self.domain_classifier(x)
        loss_adv = loss_fn(domain_pred, domain_label.unsqueeze(1).float())
        return loss_adv
    
    def forward(self, x, adapt = False, source= True, constant=1., mode=None):
        batch_size = x.size(0)
        point_num = x.size(2)
        
        transform = self.trans_net1(x)
        x = x.transpose(2, 1)
        x = x.squeeze()
        x = torch.bmm(x, transform)
        x = x.unsqueeze(3)
        x = x.transpose(2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        transform = self.trans_net2(x)
        x = x.transpose(2, 1)
        x = x.squeeze()
        x = torch.bmm(x, transform)
        x = x.unsqueeze(3)
        x = x.transpose(2, 1)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x, _ = torch.max(x, dim=2, keepdim=False)
        x = x.squeeze()#batchsize*1024
        if adapt == True:
            mid_feature = x
        x = self.mlp1(x)#batchsize*512
        x = self.dropout1(x)
        x = self.mlp2(x)#batchsize*256
        x = self.dropout2(x)
        class_out = self.class_cls(x)#batchsize*10
        
        loss_adv = self.get_adversarial_result(x, source, constant)
        
        if self.training:
            if mode=='cls':
                return class_out
            return class_out, loss_adv, x
        else:
            return class_out