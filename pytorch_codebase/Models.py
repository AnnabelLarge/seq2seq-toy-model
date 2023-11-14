#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2023

@author: annabel

Make a pytorch version of the model file, so I can compare this against the
  jax version of the codebase

Based on the implementation I did in CS288
"""
import torch
from torch import nn


class Seq2seqLSTMs(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bidirection_enc: bool, 
                 num_layers: int, dropout: float = 0.0):
        super(Seq2seqLSTMs, self).__init__()
        
        ### layers for encoder half
        self.enc_lstm = nn.LSTM(input_size=input_size, 
                               hidden_size=hidden_size, 
                               num_layers=num_layers, 
                               batch_first=True, 
                               dropout=dropout, 
                               bidirectional=bidirection_enc)
        self.bidirection_enc = bidirection_enc
        self.num_layers = num_layers
        
        ### layers for decoder half
        self.dec_lstm = nn.LSTM(input_size=input_size, 
                               hidden_size=hidden_size, 
                               num_layers=num_layers, 
                               batch_first=True, 
                               dropout=dropout, 
                               bidirectional=False)
        self.out_lin = nn.Linear(hidden_size, input_size)
        
        ### layers for both
        self.final_dropout = nn.Dropout(p=dropout)
        self.loss_fn = nn.MSELoss(reduction='mean')
    
    
    def encode(self, source):
        # artificially add a third dimension to source
        source = source.unsqueeze(2)
        
        # forward pass through encoder
        enc_out, enc_hidden = self.enc_lstm(source)
        
        # if the model is bidirectional, the first dimension will be 
        # 2 * num_layers; add the two hidden vectors
        if self.bidirection_enc:
            hn_added = (enc_hidden[0][0:self.num_layers,:,:] + 
                        enc_hidden[0][self.num_layers:,:,:])
            cn_added = (enc_hidden[1][0:self.num_layers,:,:] + 
                        enc_hidden[1][self.num_layers:,:,:])
            enc_hidden = (hn_added, cn_added)
        
        return (enc_out, enc_hidden)
    
    
    def decode(self, decoder_input, initial_hidden):
        # artificially add a third dimension to decoder input
        decoder_input = decoder_input.unsqueeze(2)
        
        # forward pass through decoder
        dec_out, dec_hidden = self.dec_lstm(decoder_input, initial_hidden)
        
        # final dropout + linear prediction layer
        # might need a final activation, to clamp output... add that only if
        #   really necessary
        dec_out = self.final_dropout(dec_out)
        final_prediction = self.out_lin(dec_out).squeeze(-1)
        return (final_prediction, dec_hidden)
    
    
    def compute_loss(self, source, target):
        """
        used explicitly for training
        this could actually be outside the model definition... whatever
        """
        ### First, encode the source and get the hidden representations
        _, enc_hidden = self.encode(source)
    
        ### Second, decode
        # input to decoder should be 
        # (last element from source + all except last element from target)
        teacher_forcing_input = torch.cat([source[:,-1].unsqueeze(1), 
                                           target[:,:-1]], dim=1)
        
        # decode
        prediction, decoder_hidden = self.decode(teacher_forcing_input, 
                                                 enc_hidden)
    
        ### Finally, the loss (MSE, in this case)
        batch_loss = self.loss_fn(prediction, target)
    
        return batch_loss



    