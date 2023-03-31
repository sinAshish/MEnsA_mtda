import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.networks import (CALayer, 
                      GradReverse,
                      conv_2d,
                      adapt_layer_off,
                      transform_net,
                      Discriminator
                      )

# Generator
class Pointnet_g(nn.Module):
    def __init__(self):
        super(Pointnet_g, self).__init__()
        self.trans_net1 = transform_net(3, 3)
        self.trans_net2 = transform_net(64, 64)
        self.conv1 = conv_2d(3, 64, 1)
        self.conv2 = conv_2d(64, 64, 1)
        # SA Node Module
        self.conv3 = adapt_layer_off()  # (64->128)
        self.conv4 = conv_2d(128, 128, 1)
        self.conv5 = conv_2d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(1024)

    def forward(self, x, node = False):
        x_loc = x.squeeze(-1)

        transform = self.trans_net1(x)
        x = x.transpose(2, 1)

        x = x.squeeze(-1)
        x = torch.bmm(x, transform)
        x = x.unsqueeze(3)
        x = x.transpose(2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        transform = self.trans_net2(x)
        x = x.transpose(2, 1)

        x = x.squeeze(-1)
        x = torch.bmm(x, transform)
        x = x.unsqueeze(3)
        x = x.transpose(2, 1)

        x, node_fea, node_off = self.conv3(x, x_loc)  # x = [B, dim, num_node, 1]/[64, 64, 1024, 1]; x_loc = [B, xyz, num_node] / [64, 3, 1024]
        x = self.conv4(x)
        x = self.conv5(x)

        x, _ = torch.max(x, dim=2, keepdim=False)

        x = x.squeeze(-1)
  
        x = self.bn1(x)

        if node == True:
            return x, node_fea, node_off
        else:
            return x, node_fea

# Classifier
class Pointnet_c(nn.Module):
    def __init__(self, in_dim=1024,num_class=10):
        super(Pointnet_c, self).__init__()
        self.fc = nn.Linear(in_dim, num_class)
        
    def forward(self, x):
        x = self.fc(x)
        return x

class Discriminator1(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256):
        super(Discriminator1, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dis1 = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dis2 = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        x = F.relu(self.dis1(x))
        x = self.bn(x)
        x = self.dis2(x)
        return x
    
class Net_MDA(nn.Module):
    def __init__(self, model_name='mda', num_classes=10):
        super(Net_MDA, self).__init__()
        self.model_name = model_name
        self.g = Pointnet_g() 
        self.class_cls = Pointnet_c(num_class= num_classes)  
        self.domain_classifier = Discriminator(1024, 4096)
        self.domain_classifier1 = Discriminator1(1024, 4096)
        
        if model_name == 'mda':
            self.attention_s = CALayer(64*64)
            self.attention_t = CALayer(64*64)
            
            self.fc1 = nn.Sequential(
                nn.Linear(4096, 1024),
                nn.ReLU()
            )
            self.fc2 = nn.Sequential(
                nn.Linear(1024*2, 4096),
                nn.ReLU(),
                nn.Linear(4096, 1024),
                nn.ReLU()
            )
            
    # Domain confusion loss
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
    
    def forward(self, x, constant=1, adaptation=False, node_vis=False, mid_feat=False, node_adaptation_s=False, node_adaptation_t=False, source=True, mode=None):
        x, feat_ori, node_idx = self.g(x, node=True)
        batch_size = feat_ori.size(0)

        # sa node visualization
        if node_vis ==True:
            return node_idx

        # collect mid-level feat
        if mid_feat == True:
            return x, feat_ori
        
        if self.model_name == 'Pointnet':
            class_out = self.class_cls(x)
            #node_class_out = self.node_cls(feat)
            loss_adv = self.get_adversarial_result(x, source, constant)
            if self.training:
                return class_out, loss_adv
            else:
                return class_out
        
        elif self.model_name =='mda':
            if source:
                # source domain sa node feat
                feat_node = feat_ori.view(batch_size, -1)
                feat = self.attention_s(feat_node.unsqueeze(2).unsqueeze(3))
                #return feat_node_s
            else:
                # target domain sa node feat
                feat_node = feat_ori.view(batch_size, -1)
                feat = self.attention_t(feat_node.unsqueeze(2).unsqueeze(3))
                #return feat_node_t
            feat = self.fc1(feat)
            x1 = torch.cat([x, feat],1)
            x1 = self.fc2(x1)
            class_out = self.class_cls(x1)
            loss_adv = self.get_adversarial_result(x1, source, constant)
            
            if self.training:
                if mode == 'cls':
                    return class_out
                return class_out, loss_adv, x1
            else:
                return class_out


if __name__ == '__main__':
    x = torch.rand((16, 1024, 3, 1))
    model = Net_MDA('mda')
    out = model(x)
