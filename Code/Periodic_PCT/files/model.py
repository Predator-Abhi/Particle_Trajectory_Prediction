import numpy as np
import torch
import torch.nn as nn
from files.graph_conv_block import Graph_Conv_Block
from files.seq2seq import Seq2Seq

class Model(nn.Module):
    def __init__(self, particles, conv_in_channels, conv_out_channels, conv_kernel_size,
                 seq_how_many, seq_hidden_size, seq_decoder_input_size, seq_num_layers, seq_output_size, seq_timesteps, conv_stride = 1, 
                 conv_dropout=0, conv_residual=True, seq_encoder_dropout=0.0, seq_decoder_dropout=0.5, 
                 seq_decoder_isDropout=True):
        
        super(Model, self).__init__()
        
        # Initialization for graph convolutional block
        self.PARTICLES                   = particles
        self.CONV_IN_CHANNELS            = conv_in_channels
        self.CONV_OUT_CHANNELS           = conv_out_channels
        self.CONV_KERNEL_SIZE            = conv_kernel_size
        self.CONV_STRIDE                 = conv_stride
        self.CONV_DROPOUT                = conv_dropout
        self.CONV_RESIDUAL               = conv_residual
        
        # Initialization for trajectory prediction block
        self.SEQ_HOW_MANY                = seq_how_many
        self.SEQ_HIDDEN_SIZE             = seq_hidden_size
        self.SEQ_DECODER_INPUT_SIZE      = seq_decoder_input_size
        self.SEQ_NUM_LAYERS              = seq_num_layers
        self.SEQ_OUTPUT_SIZE             = seq_output_size
        self.SEQ_TIMESTEPS               = seq_timesteps
        self.SEQ_ENCODER_DROPOUT         = seq_encoder_dropout
        self.SEQ_DECODER_DROPOUT         = seq_decoder_dropout
        self.SEQ_DECODER_ISDROPOUT       = seq_decoder_isDropout
        self.DEVICE                      = ("cuda" if torch.cuda.is_available() else "cpu")
        #self.DEVICE                      = "cpu"
        
        # Initialize the blocks
        self.gcb_1 = Graph_Conv_Block(particles=self.PARTICLES, in_channels=self.CONV_IN_CHANNELS, 
                                      out_channels=self.CONV_OUT_CHANNELS, kernel_size=self.CONV_KERNEL_SIZE,
                                      stride=self.CONV_STRIDE, dropout=self.CONV_DROPOUT, 
                                      first_conv=True, residual=self.CONV_RESIDUAL)
        
        self.gcb_2 = Graph_Conv_Block(particles=self.PARTICLES, in_channels=self.CONV_OUT_CHANNELS, 
                                      out_channels=self.CONV_OUT_CHANNELS, kernel_size=self.CONV_KERNEL_SIZE,
                                      stride=self.CONV_STRIDE, dropout=self.CONV_DROPOUT, 
                                      first_conv=False, residual=self.CONV_RESIDUAL)
        
        self.gcb_3 = Graph_Conv_Block(particles=self.PARTICLES, in_channels=self.CONV_OUT_CHANNELS, 
                                      out_channels=self.CONV_OUT_CHANNELS, kernel_size=self.CONV_KERNEL_SIZE,
                                      stride=self.CONV_STRIDE, dropout=self.CONV_DROPOUT, 
                                      first_conv=False, residual=self.CONV_RESIDUAL)
      
        self.tps = nn.ModuleList()
        
        for i in range(self.SEQ_HOW_MANY):
            self.tps.append(Seq2Seq(input_size=self.CONV_OUT_CHANNELS, hidden_size=self.SEQ_HIDDEN_SIZE,
                             decoder_input_size = self.SEQ_DECODER_INPUT_SIZE, output_size=self.SEQ_OUTPUT_SIZE, num_layers=self.SEQ_NUM_LAYERS, 
                             timesteps=self.SEQ_TIMESTEPS, encoder_dropout=self.SEQ_ENCODER_DROPOUT,
                             decoder_dropout=self.SEQ_DECODER_DROPOUT, 
                             decoder_isDropout=self.SEQ_DECODER_ISDROPOUT))
        
    
    def forward(self, x, G_fixed, last_location):
        assert len(last_location.shape) == 3
        
        x = self.gcb_1(x, G_fixed)
        x = self.gcb_2(x, G_fixed)
        x = self.gcb_3(x, G_fixed)
        x = torch.permute(x, (0, 3, 2, 1))
        (n, p, t, c) = x.shape
        x = x.reshape(-1, t, c)
        (n, p, c) = last_location.shape
        last_location = last_location.reshape(-1, c) 
        
        seq_outputs = []
        for i in range(self.SEQ_HOW_MANY):
            seq_outputs.append(self.tps[i](x, last_location))

        output = torch.mean(torch.stack(seq_outputs), axis=0).reshape(n, p, self.SEQ_TIMESTEPS,
                                                                 self.CONV_IN_CHANNELS)

        return output