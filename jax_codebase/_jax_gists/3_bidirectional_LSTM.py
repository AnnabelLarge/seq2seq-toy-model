#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 17:41:59 2023

@author: annabel

Create a bidirectional LSTM, and run it on some inputs
"""
from flax import linen as nn
import jax
import jax.numpy as jnp
from jax import random
from typing import Callable


#########################
### 1.) BUILD A MODEL   #
#########################
### helper functions
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


### one block
class bidirLSTM_Layer(nn.Module):
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


### the larger model
class bidirLSTM(nn.Module):
    ### place parameters here needed to initialize the module
    hidden_size: int
    n_layers: int
    dropout_prob: float
    merge_fn: Callable
    
    @nn.compact
    def __call__(self, x, training: bool=False):
        ### 1.) Artificially give the LSTM a third dimension
        # for real sequences, third dimension will be created after 
        #   initial embedding; for this toy example, just expand it
        x = jnp.expand_dims(x, axis=2)
        
        
        ### 2.) LSTM forward pass
        # first LSTM layer, so let jax initialize the carry for you
        carry, x = bidirLSTM_Layer(feats = self.hidden_size,
                                   layer_name = 'LSTM_layer0',
                                   merge_fn = self.merge_fn)(inputs=x)
        
        
        ### 3.) Optional dropout
        x = nn.Dropout(rate = self.dropout_prob, 
                       deterministic = (not training))(x)
        
        
        ### 4.) if there's any remaining layers, add those blocks on
        # the carry will be fed in from the previous LSTM layer
        # the carry ALSO needs to be a tuple of tuples
        for layer_idx in range(1, self.n_layers):
            carry, x = bidirLSTM_Layer(feats = self.hidden_size,
                                       layer_name = f'LSTM_layer{layer_idx}',
                                       merge_fn = self.merge_fn)(inputs = x,
                                                                 carry = carry)
            x = nn.Dropout(rate = self.dropout_prob, 
                           deterministic = (not training))(x)
        
        ### 5.) unpack the carry and apply some function to merge the forward 
        #       and backwards RNN carries; pulled this directly from flax
        #       source code
        carry_fw, carry_rv = carry
        carry_0_concat = jax.tree_map(self.merge_fn, carry_fw[0], carry_rv[0])
        carry_1_concat = jax.tree_map(self.merge_fn, carry_fw[1], carry_rv[1])
        carry = (carry_0_concat, carry_1_concat)
        
        ### 5.) return the carry and the value
        return (carry, x)


#####################################
### 2.) BUILD AND INITIALIZE MODEL  #
#####################################
### instantiate model
mymodel = bidirLSTM(hidden_size = 10,
                    n_layers = 2,
                    dropout_prob = 0.05,
                    merge_fn = _sum)

### initialize variables
batch_size = 2
max_seqlen = 3

# wrap this in a function to hit it with that jitterbug
# DON'T put into training mode when initializing the parameters
@jax.jit
def initialize_variables(init_rng):
    fake_x = jnp.empty((batch_size, max_seqlen))
    init_variables = mymodel.init(rngs = init_rng, 
                                  x = fake_x,
                                  training = False)
    return init_variables

init_variables = initialize_variables(init_rng = random.key(0))


#####################
### 3.) RUN MODEL   #
#####################
real_x = jnp.array([[1,2,3],
                    [4,5,0]])

# could also wrap .apply() in a jit, if you wanted to apply it to many inputs
out_carry, y = mymodel.apply(variables = init_variables,
                             x = real_x,
                             training=True, 
                             rngs={'dropout': random.key(1)})
