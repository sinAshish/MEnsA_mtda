# Part of the code borrowed from https://github.com/canqin001/PointDAN

import torch
import torch.nn as nn
import torch.nn.functional as F

import point_utils as utils

class Discriminator(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256):
        super(Discriminator, self).__init__()
        """
        Discriminator to distinguish source domain from target domains.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dis1 = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dis2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.dis1(x))
        x = self.bn(x)
        x = self.dis2(x)
        x = torch.sigmoid(x)
        return x
    
class conv_2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, activation='relu'):
        """
        Wrapper for 2D Convolution, BatchNorm and Activation
        """
        super(conv_2d, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel)
        self.bn = nn.BatchNorm2d(out_ch)
        
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'leakyrelu':
            self.act = nn.LeakyReLU()
    def forward(self, x):
        #import pdb; pdb.set_trace()
        x = self.conv(x)
        x= self.bn(x)
        x = self.act(x)
        return x

class fc_layer(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, activation='leakyrelu'):
        """
        Wrapper for Dense Layer, BatchNorm and Activation
        """
        super(fc_layer, self).__init__()
        if activation == 'relu':
            self.ac = nn.ReLU(inplace=True)
        elif activation == 'leakyrelu':
            self.ac = nn.LeakyReLU()
        self.fc = nn.Linear(in_ch, out_ch)
        if bn:
            self.bnorm = nn.BatchNorm1d(out_ch)
        self.bn = bn
    def forward(self, x):
        #import pdb; pdb.set_trace()
        x = self.fc(x)
        
        if self.bn:
            x = self.bnorm(x)
        x = self.ac(x)
        return x

class transform_net(nn.Module):
    def __init__(self, in_ch, K=3):
        """
        Transformation Network for point cloud
        Input: 
            point cloud : B x 3 x N x 1
        """
        super(transform_net, self).__init__()
        self.K = K
        self.conv2d1 = conv_2d(in_ch, 64, 1)
        self.conv2d2 = conv_2d(64, 128, 1)
        self.conv2d3 = conv_2d(128, 1024, 1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(512, 1))
        self.fc1 = fc_layer(1024, 512)
        self.fc2 = fc_layer(512, 256)
        self.fc3 = nn.Linear(256, K*K)
        
    def forward(self, x):
        #import pdb; pdb.set_trace()
        device = x.device
        x = self.conv2d1(x)
        x = self.conv2d2(x)
        x = self.conv2d3(x)
        x, _ = torch.max(x, dim=2, keepdim=False)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        iden = torch.eye(self.K).view(1,self.K * self. K).repeat(x.size(0),1)
        iden = iden.to(device)
        x = x + iden
        x = x.view(x.size(0), self.K, self.K)
        return x

# Adaptive Node attention module
class adapt_layer_off(nn.Module):
    def __init__(self, num_node=64, offset_dim=3, trans_dim_in=64, trans_dim_out=64, fc_dim=64):
        """
        Node adaptive attention layer as proposed in PointDAN
        Gives attention to local geometry of the object
        """
        super(adapt_layer_off, self).__init__()
        self.num_node = num_node
        self.offset_dim = offset_dim
        self.trans = conv_2d(trans_dim_in, trans_dim_out, 1)
        self.pred_offset = nn.Sequential(
            nn.Conv2d(trans_dim_out, offset_dim, kernel_size=1, bias=False),
            nn.Tanh())
        self.residual = conv_2d(trans_dim_in, fc_dim, 1)

    def forward(self, input_fea, input_loc):
        # Initialize node
        #import pdb; pdb.set_trace()
        fpoint_idx = utils.farthest_point_sample(input_loc, self.num_node)  # (B, num_node)
        fpoint_loc = utils.index_points(input_loc, fpoint_idx)  # (B, 3, num_node)
        fpoint_fea = utils.index_points(input_fea, fpoint_idx)  # (B, C, num_node)
        group_idx = utils.query_ball_point(0.3, 64, input_loc, fpoint_loc)   # (B, num_node, 64)
        group_fea = utils.index_points(input_fea, group_idx)  # (B, C, num_node, 64)
        group_fea = group_fea - fpoint_fea.unsqueeze(3).expand(-1, -1, -1, self.num_node)

        # Learn node offset
        seman_trans = self.pred_offset(group_fea)  # (B, 3, num_node, 64)
        group_loc = utils.index_points(input_loc, group_idx)   # (B, 3, num_node, 64)
        group_loc = group_loc - fpoint_loc.unsqueeze(3).expand(-1, -1, -1, self.num_node)
        node_offset = (seman_trans*group_loc).mean(dim=-1)

        # Update node and get node feature
        node_loc = fpoint_loc+node_offset.squeeze(-1)  # (B,3,num_node)
        group_idx = utils.query_ball_point(None, 64, input_loc, node_loc)
        residual_fea = self.residual(input_fea)
        group_fea = utils.index_points(residual_fea, group_idx)
        node_fea, _ = torch.max(group_fea, dim=-1, keepdim=True)

        # Interpolated back to original point
        output_fea = utils.upsample_inter(input_loc, node_loc, input_fea, node_fea, k=3).unsqueeze(3)

        return output_fea, node_fea, node_offset

# Channel Attention
class CALayer(nn.Module):
    def __init__(self, channel, reduction=8):
        """
        Channel Attention Layer
        """
        super(CALayer, self).__init__()
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
        self.bn = nn.BatchNorm1d(4096)

    def forward(self, x):
        y = self.conv_du(x)
        y = x * y + x
        y = y.view(y.shape[0], -1)
        y =self.bn(y)
        
        return y

# Gradient  Reversal
class GradReverse(torch.autograd.Function):
    """
    Gradient Reversal Layer
    Takes the gradient and reverses it by multiplying with a negative scaler.
    Promotes adverserial domain invariant learning. 
    """
    def __init__(self, lambd):
        self.lambd = lambd
        
    @staticmethod
    def forward(ctx, x, lambd):
        result =  x.view_as(x)
        ctx.lambda_ = lambd
        return result 
    @staticmethod
    def backward(ctx, grad_output):
        ##import pdb; pdb.set_trace()
        out = grad_output.neg() * ctx.lambda_
        return out, None