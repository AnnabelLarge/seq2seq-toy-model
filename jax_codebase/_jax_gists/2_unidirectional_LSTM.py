#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 17:41:59 2023

@author: annabel

Create a unidirectional LSTM, and run it on some inputs
"""
from flax import linen as nn
import jax
import jax.numpy as jnp
from jax import random


#########################
### 1.) BUILD A MODEL   #
#########################
### one layer
class unidirLSTM_Layer(nn.Module):
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


### the larger model
class unidirLSTM(nn.Module):
    ### place parameters here needed to initialize the module
    hidden_size: int
    n_layers: int
    dropout_prob: float
    
    @nn.compact
    def __call__(self, x, training: bool=False):
        ### 1.) Artificially give the LSTM a third dimension
        # for real sequences, third dimension will be created after 
        #   initial embedding; for this toy example, just expand it
        x = jnp.expand_dims(x, axis=2)
        
        
        ### 2.) LSTM forward pass
        # first LSTM layer, so let jax initialize the carry for you
        carry, x = unidirLSTM_Layer(feats = self.hidden_size,
                                    layer_name = 'LSTM_layer0')(inputs=x)
        
        
        ### 3.) Optional dropout
        x = nn.Dropout(rate = self.dropout_prob, 
                       deterministic = (not training))(x)
        
        
        ### 4.) if there's any remaining layers, add those blocks on
        # the carry will be fed in from the previous LSTM layer
        for layer_idx in range(1, self.n_layers):
            carry, x = unidirLSTM_Layer(feats = self.hidden_size,
                                        layer_name = f'LSTM_layer{layer_idx}')(inputs = x,
                                                                               carry = carry)
            x = nn.Dropout(rate = self.dropout_prob, 
                           deterministic = (not training))(x)
        
        
        ### 5.) return the carry and the value
        return (carry, x)


###########################
### 2.) INITIALIZE MODEL  #
###########################
### instantiate model
mymodel = unidirLSTM(hidden_size = 6,
                     n_layers = 4,
                     dropout_prob = 0.05)


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
                    [4,5,6]])


# could also wrap .apply() in a jit, if you wanted to apply it to many inputs
carry, y = mymodel.apply(variables = init_variables,
                             x = real_x,
                             training=True, 
                             rngs={'dropout': random.key(1)})
