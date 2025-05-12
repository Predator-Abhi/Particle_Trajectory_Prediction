import numpy as np
import torch
import torch.nn as nn
from files.graph_operation_layer import GraphOperation

class Graph_Conv_Block(nn.Module):
    def __init__(self, particles, in_channels, out_channels, kernel_size,
                 stride=1, dropout=0, first_conv = False, residual=True):
        super(Graph_Conv_Block, self).__init__()
        self.DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")
        #self.DEVICE = "cpu"
        
        assert len(kernel_size) == 2
        assert kernel_size[0]%2 == 1
        padding = ((kernel_size[0]-1)//2, 0)
        
        # First convolution
        if first_conv:
            self.first_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1), device=self.DEVICE),
                nn.BatchNorm2d(out_channels, device=self.DEVICE)
            )
            if residual:
                self.residual = lambda x: x
                
        else:
            self.first_conv = lambda x: x
            
        # Adding residual based on where it is
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
            
        # Graph operation
        self.go = GraphOperation(particles)
        
        # Temporal convolution block
        self.tc = nn.Sequential(
            nn.BatchNorm2d(out_channels, device=self.DEVICE),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size, (stride, 1), padding, device=self.DEVICE),
            nn.BatchNorm2d(out_channels, device=self.DEVICE),
            nn.Dropout(dropout, inplace=False),
        )     
        
        self.relu = nn.ReLU(inplace=False)
        
    def forward(self, x, G_fixed):
        x = self.first_conv(x)
        res = self.residual(x)
        x = self.go(x, G_fixed)
        x = self.tc(x) + res
        return self.relu(x)