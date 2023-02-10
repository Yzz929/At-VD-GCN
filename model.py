import os
import math
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from TCN import TemporalConvNet
import torch.optim as optim
# from .attention import CBAM, SE, ECA  #导入注意力机制模块

class abc(nn.Module):
    def __init__(self, c1, c2):
        super(abc, self).__init__()
        self.conv1 = nn.Conv2d(c1, c2, kernel_size=(3,1), stride=1,padding=(1,0))
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=(5,1), stride=1,padding=(2,0))
        self.conv3 = nn.Conv2d(c2 * 3, c2, kernel_size=(1,1),stride=1,padding=(0,0))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        y = torch.cat((x, x1, x2), dim=1)
        return self.conv3(y)

class SE(nn.Module):
    def __init__(self, channel, ratio=16):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // ratio, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // ratio, channel, bias=False),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ECA(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(ECA, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        print("kernel_size:",kernel_size)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, None))
        self.max_pool = nn.AdaptiveMaxPool2d((1, None))

        self.fc1 = nn.Conv2d(in_planes, in_planes // 4, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 4, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # x = self.avg_pool(x)
        # x = self.fc1(x)
        # x = self.relu1(x)
        # avg_out = self.fc2(x)

        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # x = self.max_pool(x)
        # x = self.fc1(x)
        # x = self.relu1(x)
        # max_out = self.fc2(x)

        out = avg_out + max_out
        return self.sigmoid(out)

class ConvTemporalGraphical(nn.Module):
    #Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    r"""The basic module for applying a graph convolution.应用图卷积的基本模块
    Args:
        in_channels (int): Number of channels in the input sequence data 输入序列数据中的通道数
        out_channels (int): Number of channels produced by the convolution 卷积产生的通道数
        kernel_size (int): Size of the graph convolving kernel图卷积核的大小
        t_kernel_size (int): Size of the temporal convolving kernel时间卷积核的大小
        t_stride (int, optional): Stride of the temporal convolution. Default: 1时间卷积的步幅
        t_padding (int, optional): Temporal zero-padding added to both sides of 向输入的两边添加 时间零填充
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.时间核元素之间的间隔
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.在输出中添加一个可学习的偏差
            Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,批量尺寸
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,空间内核大小
            :math:`T_{in}/T_{out}` is a length of input/output sequence,输入/输出序列的长度
            :math:`V` is the number of graph nodes. 图形节点的数量
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,#输入的图卷积核大小
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super(ConvTemporalGraphical,self).__init__()#super在子类中需要调用父类
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

        # self.se = SE(int(out_channels))
        self.CA = ChannelAttention(out_channels)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size #size() 0：返回该二维矩阵的行数 1：返回该二维矩阵的列数
        x = self.conv(x)
        x = self.CA(x)
        x = torch.einsum('nctv,tvw->nctw', (x, A))
        return x.contiguous(), A
    

class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.在一个输入图序列上应用空间-时间图卷积
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution卷积产生的通道数
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``应用残差机制
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,#2
                 out_channels,#5
                 kernel_size,
                 use_mdn = False,
                 stride=1,
                 dropout=0,
                 residual=True):
        super(st_gcn,self).__init__()
        
        print("outstg",out_channels)

        assert len(kernel_size) == 2 #时间卷积8与图卷积3
        assert kernel_size[0] % 2 == 1#图卷积长度为奇数
        # padding = ((kernel_size[0] - 1) // 2, 0)

        self.use_mdn = use_mdn
        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])
        # self.conv1 = nn.Conv2d (out_channels ,out_channels ,kernel_size=(1,3),stride= 1,padding=(0,1))
        # self.conv2 = nn.Conv2d (out_channels ,out_channels ,kernel_size=(1,5),stride= 1,padding=(0,2))
        # self.conv3 = nn.Conv2d (out_channels * 3,out_channels ,kernel_size= (1,1))

        #self.tcn = TemporalConvNet(out_channels,[8,8,8,8])#
        self.tcn = nn.Sequential( #模块将按照构造函数中传递的顺序添加到模块中
            nn.BatchNorm2d(out_channels), #num_features：一般输入参数为batch_size*num_features*height*width，即为其中特征的数量
            nn.PReLU(),
            # abc(out_channels,out_channels),

            # nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), (stride, 1), padding,),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        # self.se = SE(int(out_channels))
        # self.eca = ECA(int(out_channels))

        if not residual: #residual=ture
            self.residual = lambda x: 0 #lambda argument_list: expression 输入是传入到参数列表argument_list的值，输出是根据表达式expression计算得到的值

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.prelu = nn.PReLU()

    def forward(self, x, A):# x, A 输入

        res = self.residual(x)
        x, A = self.gcn(x, A)
        # x = x[:, :, :, 0]
        # x1 = self.conv1(x)
        # x2 = self.conv1(x1)
        # x3 = torch.cat((x,x1,x2),dim=1)
        # x = self.conv3(x3)

        # x = self.tcn(x)#,[8,8,8,8]
        x = self.tcn(x)+ res
        # x =self.se(x) + res
        # x =self.eca(x) + res

        if not self.use_mdn:
            x = self.prelu(x)

        return x, A

# class ts_stgcn(nn.Module):
#
#     def __init__(self, in_channels, out_channels, kernel_size):
#         super().__init__()
#
#         self.origin_stream = st_gcn(in_channels, out_channels, kernel_size)
#         self.motion_stream = st_gcn(in_channels, out_channels, kernel_size)
#
#     def forward(self, x):
#         N, C, T, V, M = x.size()
#         m = torch.cat((torch.FloatTensor(N, C, 1, V, M).zero_(),
#                         x[:, :, 1:-1] - 0.5 * x[:, :, 2:] - 0.5 * x[:, :, :-2],
#                         torch.FloatTensor(N, C, 1, V, M).zero_()), dim=2)
#
#         res = self.origin_stream(x) + self.motion_stream(m)
#         return res




class social_stgcnn(nn.Module):
    def __init__(self,n_stgcnn =1,n_txpcnn=1,input_feat=2,output_feat=5,#1个STG CNN  1个TXP CNN
                 seq_len=8,pred_seq_len=12,kernel_size=3):
        super(social_stgcnn,self).__init__()
        self.n_stgcnn = n_stgcnn
        self.n_txpcnn = n_txpcnn
                
        self.st_gcns = nn.ModuleList()
        self.st_gcns.append(st_gcn(input_feat,output_feat,(kernel_size,seq_len)))

        for j in range(1,self.n_stgcnn):
            self.st_gcns.append(st_gcn(output_feat,output_feat,(kernel_size,seq_len)))


        self.tpcnns = nn.ModuleList()
        self.tpcnns.append(nn.Conv2d(seq_len,pred_seq_len,kernel_size=3,padding=1))
        # self.tpcnns.append(SE(pred_seq_len))
        for j in range(1,self.n_txpcnn):
            self.tpcnns.append(nn.Conv2d(pred_seq_len,pred_seq_len,kernel_size=3,padding=1))
        self.tpcnn_ouput = nn.Conv2d(pred_seq_len,pred_seq_len,kernel_size=3,padding=1)
        # self.tpcnns.append(SE(pred_seq_len))
        self.prelus = nn.ModuleList()
        for j in range(self.n_txpcnn):
            self.prelus.append(nn.PReLU())

        
    def forward(self,v,a):

        for k in range(self.n_stgcnn):
            v,a = self.st_gcns[k](v,a)
            
        v = v.view(v.shape[0],v.shape[2],v.shape[1],v.shape[3])#用.view做一个resize的功能，用.shape来进行选择通道数
        
        v = self.prelus[0](self.tpcnns[0](v))

        for k in range(1,self.n_txpcnn-1):
            v = self.prelus[k](self.tpcnns[k](v)) + v
            
        v = self.tpcnn_ouput(v)
        v = v.view(v.shape[0],v.shape[2],v.shape[1],v.shape[3])


        return v,a
