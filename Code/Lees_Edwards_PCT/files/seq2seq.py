import numpy
import torch
import torch.nn as nn

class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, decoder_input_size, output_size, num_layers, timesteps, 
                 encoder_dropout=0.0, decoder_dropout=0.5, decoder_isDropout=True):
        super(Seq2Seq, self).__init__()
        self.INPUT_SIZE = input_size
        self.HIDDEN_SIZE = hidden_size
        self.DECODER_INPUT_SIZE = decoder_input_size
        self.OUTPUT_SIZE = output_size
        self.NUM_LAYERS = num_layers
        self.TIME_STEPS = timesteps
        self.ENCODER_DROPOUT = encoder_dropout
        self.DECODER_DROPOUT = decoder_dropout
        self.DECODER_ISDROPOUT = decoder_isDropout
        self.DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")
        #self.DEVICE = "cpu"
        
        # creating encoder and decoder sequence
        self.encoder_sequence = nn.GRU(self.INPUT_SIZE, self.HIDDEN_SIZE, self.NUM_LAYERS,
                                       dropout=self.ENCODER_DROPOUT, batch_first=True, device=self.DEVICE)
        self.decoder_sequence = nn.ModuleList()

        for i in range(self.NUM_LAYERS):
            if i==0:
                self.decoder_sequence.append(nn.GRUCell(self.DECODER_INPUT_SIZE, self.HIDDEN_SIZE, device=self.DEVICE))
            if i>0:
                self.decoder_sequence.append(nn.GRUCell(self.HIDDEN_SIZE, self.HIDDEN_SIZE, device=self.DEVICE))
    
        self.linear = nn.Linear(self.HIDDEN_SIZE, self.OUTPUT_SIZE, device=self.DEVICE)
        self.dropout_layer = nn.Dropout(self.DECODER_DROPOUT)
    
    def encoder_gru(self, in_data):
        assert len(in_data.shape) == 3
        _, hidden = self.encoder_sequence(in_data)
        return hidden
    
    def decoder_gru(self, hidden, last_location):
        assert len(last_location.shape) == 2
        assert len(hidden.shape) == 3 
        
        outputs = []                   
        hidden_dec_curr = hidden
        for i in range(1, self.TIME_STEPS+1):
            curr_cell_x = last_location
            dev = last_location.device
            hidden_dec_next = torch.zeros((self.NUM_LAYERS, curr_cell_x.shape[0], self.HIDDEN_SIZE), device=dev)
            
            for j in range(self.NUM_LAYERS):
                curr_cell_x = self.decoder_sequence[j](curr_cell_x, hidden_dec_curr[j])
                hidden_dec_next[j] = curr_cell_x

            # Applying dropout
            if self.DECODER_ISDROPOUT:
                gru_out = self.dropout_layer(hidden_dec_next[-1])
            else:
                gru_out = hidden_dec_next[-1]
            # Adding deltas (dx) with original location x
            linear_out = self.linear(gru_out)
            
            last_location = linear_out + last_location
            outputs.append(last_location.unsqueeze(axis=1))
            
            hidden_dec_curr = hidden_dec_next
            
        outputs = torch.cat(outputs, axis=1)
        return outputs
    
    def forward(self, in_data, last_location):
        hidden = self.encoder_gru(in_data)
        outputs = self.decoder_gru(hidden, last_location)
        return outputs