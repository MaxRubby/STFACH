import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.hetero_utils import *

class HeteroGCNs(nn.Module):
    def __init__(self,  device, num_nodes, dropout=0.3, in_dim=64,out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2, g_eps = 5, s_eps = 5, adj_mx = None,adj_semx = None,supports = None):
        super(HeteroGCNs, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.geo_gconv = nn.ModuleList()
        self.sem_gconv = nn.ModuleList()

        adj_mx =torch.tensor( floyd_warshall_optimized(adj_mx, num_nodes = num_nodes)).float()
        self.geo_mask = (adj_mx <= g_eps).float().to(device)
        self.sem_mask = torch.tensor(retain_top_k_neighbors(adj_semx, k = s_eps)).bool().float().to(device)
        self.sem_mask = (~self.sem_mask.bool()).float()
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.supports =  []
        receptive_field = 1
        self.supports_len = 0

        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)  
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
        self.nodevec3 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)  
        self.nodevec4 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
        self.supports_len +=1

        for b in range(blocks):
            additional_scope = kernel_size - 1  
            new_dilation = 1  
            for i in range(layers):
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,  
                                                   out_channels=dilation_channels, #32
                                                   kernel_size=(1,kernel_size),dilation=new_dilation)) #kernel_size 2， new_dilation 1  
                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2  
                receptive_field += additional_scope  
                additional_scope *= 2  
            
                self.geo_gconv.append(masked_gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))
                self.sem_gconv.append(masked_gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))



        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field
    def forward(self, input): # ! (B, C, N, T)
        input = input.permute(0, 3, 2, 1)
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0)) #避免空洞？所以需要填充  torch.Size([64, 2, 307, 13])
        else:
            x = input
        
        x = self.start_conv(x)  
        skip = 0
        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1) #relu 转负为正，softmax在列上进行softmax，统计每行的比例，也即每个节点与其他节点的关联性矩阵
            adp2 = F.softmax(F.relu(torch.mm(self.nodevec3, self.nodevec4)), dim=1) #relu 转负为正，softmax在列上进行softmax，统计每行的比例，也即每个节点与其他节点的关联性矩阵
            new_supports = self.supports + [adp]
            new_supports2 = self.supports + [adp2]

        # Temporal Aggregation layers
        for i in range(self.blocks * self.layers):
            residual = x  
            filter = self.filter_convs[i](residual)  
            filter = torch.tanh(filter)  
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate  
            s = x 
            s = self.skip_convs[i](s)  

            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            if self.supports is not None:
                x_geo = self.geo_gconv[i](x, new_supports, self.geo_mask)  
                x_sem = self.sem_gconv[i](x, new_supports2, self.sem_mask)  
                x = x_geo+x_sem
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        x = F.relu(skip)  
        x = F.relu(self.end_conv_1(x))  
        x = self.end_conv_2(x)  
        x = torch.sigmoid(x)
        return x
