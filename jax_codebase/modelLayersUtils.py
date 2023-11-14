#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:04:59 2023

@author: annabel

Taking inspiration from a couple different sources:
    > https://github.com/google/flax/blob/main/examples/seq2seq/models.py
      >> basic encoder-decoder setup in jax
      
    > https://github.com/google/flax/blob/main/examples/sst2/models.py
      >> provides an example of a bidirectional LSTM cell
    
  
Eventually for true seq2seq tasks, I'll need some sort of sequence embedder
"""
from flax import linen as nn
import jax
import jax.numpy as jnp
from typing import Callable


#########################
### individual layers   #
#########################
class unidirLSTM_Layer(nn.Module):
    """
    One unidirectional LSTM layer
    
    initialize with:
    -----------------
    feats: hidden_dim size
    layer_name: a unique name, to easily retrieve its parameters from paramdict
        
    
    arguments for call method:
    --------------------------
    inputs: input of size (batch, seq_len, H); H could be embedding dim, hidden
            dim, etc.
    carry: hidden state, a tuple containing (array1, array2)
           array1 and array2 both have sizes (batch, hidden_dim)
    
        
    outputs:
    --------
    out_carry: hidden state, a tuple containing (array1, array2)
               array1 and array2 both have sizes (batch, hidden_dim)
    output: output of size (batch, seq_len, hidden_dim)
    """
    feats: int
    layer_name: str
    
    @nn.compact
    def __call__(self, inputs, carry=None):
        lstmcell = nn.OptimizedLSTMCell(features=self.feats, 
                                        name=f'{self.layer_name}')
        out_carry, output = nn.RNN(cell = lstmcell, 
                                   return_carry = True)(inputs = inputs,
                                                        initial_carry = carry)
        return (out_carry, output)


class bidirLSTM_Layer(nn.Module):
    """
    One bidirectional LSTM layer
        
    initialize with:
    -----------------
    feats: hidden_dim size
    layer_name: a unique name, to easily retrieve its parameters from paramdict
    merge_fn: how to merge outputs from forward and reverse LSTM
        > _concatenate: concatenates as [forward, reverse]; D=2
            >> output shape: (batch, seq_len, 2 * hidden_dim)
        > _sum: adds forward and reverse answers
            >> output shape: (batch, seq_len, hidden_dim); D=1
        
    
    arguments for call method:
    --------------------------
    inputs: input of size (batch, seq_len,  H); H could be embedding dim, 
            hidden dim, etc.
    carry: hidden state, a nested tuple of tuples: 
           ((Array, Array), (Array, Array))
           all arrays have sizes (batch, D * hidden_dim), where concatenation
           means D=2, and summing means D = 1
    
        
    outputs:
    --------
    out_carry: hidden state, a nested tuple of tuples: 
               ((Array, Array), (Array, Array))
               all arrays have sizes (batch, D * hidden_dim), where concatenation
               means D=2, and summing means D = 1
    output: output of size (batch, seq_len, hidden_dim)
    """
    feats: int
    layer_name: str
    merge_fn: Callable
    
    @nn.compact
    def __call__(self, inputs, carry=None):
        # create the forward RNN
        lstmcell_fw = nn.OptimizedLSTMCell(features=self.feats, 
                                           name=f'{self.layer_name}_FW')
        rnn_fw = nn.RNN(cell = lstmcell_fw)
        
        
        # create the reverse RNN
        # by default, this will run with reverse=True and keep_order=True
        #   such that forward and reverse outputs will be in the same order
        lstmcell_rv = nn.OptimizedLSTMCell(features=self.feats, 
                                           name=f'{self.layer_name}_RV')
        rnn_rv = nn.RNN(cell = lstmcell_rv)
        
        # place both in a bidirectional layer
        raw_carry, output = nn.Bidirectional(forward_rnn = rnn_fw, 
                                          backward_rnn = rnn_rv,
                                          return_carry = True,
                                          merge_fn = self.merge_fn)(inputs=inputs, 
                                                                    initial_carry=carry)
                                                                       
        return (raw_carry, output)



###################################
### helper functions for biLSTM   #
###################################
def _concatenate(a, b):
    """
    used to concatenate the hidden dimensions from forward and reverse RNNs
    pulled this directly from flax source code
    """
    return jnp.concatenate([a, b], axis=-1)

def _sum(a, b):
    """
    used to add the hidden dimensions from forward and reverse RNNs
    """
    return jnp.sum(jnp.array([a,b]), axis=0)

