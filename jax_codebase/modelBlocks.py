#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 17:30:16 2023

@author: annabel

These have full encoder and decoder blocks, built from layers and functions
  found in modelUtils. Use these in the training loop themselves?

TODO: there's probably some clever way to include "bidirectional" as an
      option to the encoder, but for now, just have separate encoding
      modules

TODO: it looks like from flax module documentation, that I could declare
      encoder and decoder as separate methods? should try it out sometime
"""
from flax import linen as nn
import jax
import jax.numpy as jnp
from modelLayersUtils import *


################
### ENCODERS   #
################
### Uni-directional
class uniEncodeLSTM(nn.Module):
    """
    Uni-directional LSTM with dropout
    """
    ### place parameters here needed to initialize the module
    hidden_size: int
    n_layers: int
    dropout_prob: float
    
    @nn.compact
    def __call__(self, x_toenc, training):
        ### 1.) Artificially give the LSTM a third dimension
        # for real sequences, third dimension will be created after 
        #   initial embedding; for this toy example, just expand it
        x_toenc = jnp.expand_dims(x_toenc, axis=2)
        
        
        ### 2.) LSTM forward pass
        # first LSTM layer, so let jax initialize the carry for you
        carry_enc, x_enc = unidirLSTM_Layer(feats = self.hidden_size,
                                    layer_name = 'encodeLSTM_layer0')(inputs=x_toenc)
        
        
        ### 3.) Optional dropout
        x_enc = nn.Dropout(rate = self.dropout_prob,
                           name = 'encodeLSTM_dropout0',
                       deterministic = (not training))(x_enc)
        
        
        ### 4.) if there's any remaining layers, add those blocks on
        # the carry will be fed in from the previous LSTM layer
        for layer_idx in range(1, self.n_layers):
            carry_enc, x_enc = unidirLSTM_Layer(feats = self.hidden_size,
                            layer_name = f'encodeLSTM_layer{layer_idx}')(inputs = x_enc,
                                                                   carry = carry_enc)
            x_enc = nn.Dropout(rate = self.dropout_prob, 
                               name = f'encodeLSTM_dropout{layer_idx}',
                               deterministic = (not training))(x_enc)
        
        
        ### 5.) return the carry and the value
        return (carry_enc, x_enc)


### Bi-directional
class biEncodeLSTM(nn.Module):
    """
    Bi-directional LSTM with dropout
    """
    ### place parameters here needed to initialize the module
    hidden_size: int
    n_layers: int
    dropout_prob: float
    merge_fn: Callable
    
    @nn.compact
    def __call__(self, x_toenc, training):
        ### 1.) Artificially give the LSTM a third dimension
        # for real sequences, third dimension will be created after 
        #   initial embedding; for this toy example, just expand it
        x_toenc = jnp.expand_dims(x_toenc, axis=2)
        
        
        ### 2.) LSTM forward pass
        # first LSTM layer, so let jax initialize the carry for you
        carry_enc, x_enc = bidirLSTM_Layer(feats = self.hidden_size,
                                    layer_name = 'encodeLSTM_layer0',
                                    merge_fn = self.merge_fn)(inputs=x_toenc)
        
        
        ### 3.) Optional dropout
        x_enc = nn.Dropout(rate = self.dropout_prob, 
                           name = 'encodeLSTM_dropout0',
                           deterministic = (not training))(x_enc)
        
        
        ### 4.) if there's any remaining layers, add those blocks on
        # the carry will be fed in from the previous LSTM layer
        # the carry ALSO needs to be a tuple of tuples
        for layer_idx in range(1, self.n_layers):
            carry_enc, x_enc = bidirLSTM_Layer(feats = self.hidden_size,
                                layer_name = f'encodeLSTM_layer{layer_idx}',
                                merge_fn = self.merge_fn)(inputs = x_enc,
                                                          carry = carry_enc)
            x_enc = nn.Dropout(rate = self.dropout_prob, 
                               name = f'encodeLSTM_dropout{layer_idx}',
                           deterministic = (not training))(x_enc)
        
        ### 5.) unpack the carry and apply some function to merge the forward 
        #       and backwards RNN carries; pulled this directly from flax
        #       source code
        carry_fw, carry_rv = carry_enc
        carry_0_concat = jax.tree_map(self.merge_fn, carry_fw[0], carry_rv[0])
        carry_1_concat = jax.tree_map(self.merge_fn, carry_fw[1], carry_rv[1])
        carry_enc = (carry_0_concat, carry_1_concat)
        
        ### 5.) return the carry and the value
        return (carry_enc, x_enc)



################
### DECODERS   #
################
class DecodeLSTM(nn.Module):
    """
    Uni-directional LSTM with dropout
    """
    ### place parameters here needed to initialize the module
    hidden_size: int
    n_layers: int
    dropout_prob: float
    
    @nn.compact
    def __call__(self, x_todec, carry_todec, training):
        ### 1.) Artificially give the LSTM a third dimension
        # for real sequences, third dimension will be created after 
        #   initial embedding; for this toy example, just expand it
        x_todec = jnp.expand_dims(x_todec, axis=2)
        
        
        ### 2.) LSTM forward pass
        # first LSTM layer, so let jax initialize the carry for you
        carry_dec, x_dec = unidirLSTM_Layer(feats = self.hidden_size,
                                    layer_name = 'decodeLSTM_layer0')(inputs=x_todec,
                                                                carry=carry_todec)
        
        
        ### 3.) Optional dropout
        x_dec = nn.Dropout(rate = self.dropout_prob, 
                           name = 'decodeLSTM_dropout0',
                       deterministic = (not training))(x_dec)
        
        
        ### 4.) if there's any remaining layers, add those blocks on
        # the carry will be fed in from the previous LSTM layer
        for layer_idx in range(1, self.n_layers):
            carry_dec, x_dec = unidirLSTM_Layer(feats = self.hidden_size,
                            layer_name = f'decodeLSTM_layer{layer_idx}')(inputs = x_dec,
                                                                   carry = carry_dec)
            x_dec = nn.Dropout(rate = self.dropout_prob,
                               name = f'decodeLSTM_dropout{layer_idx}', 
                           deterministic = (not training))(x_dec)
        
        
        ### 5.) final dense layer for final prediction 
        # (batch, seqLen, hidden) -> (batch, seqLen, 1)
        x_pred = nn.Dense(features=1, name='Dense_finalPred')(x_dec)
        
        # remove redundant final axis
        x_pred = jnp.squeeze(x_pred, axis=-1)
        
        return (carry_dec, x_pred)


