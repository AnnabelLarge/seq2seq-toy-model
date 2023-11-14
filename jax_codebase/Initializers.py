#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 22:01:24 2023

@author: annabel_large

Initialize encoder and decoders, and return train states (which are models
with params and the optimizer)

Really only call this twice, so it's not worth jitting (or rather, not worth
  the troubleshooting efforts to make jitable)
"""
from flax import linen as nn
import jax
import jax.numpy as jnp
from flax.training import train_state
from jax import random
from modelLayersUtils import *
from modelBlocks import *


###################################
### initializees the train states #
###################################
### unidirectional encoder, unidirectional encoder
def unidirec_init(hidden_dim, num_layers, dropout, batch_size, inseq_maxlen,
                  outseq_maxlen, tx, rngkey, merge_fn=None):
    ### 1.) rng keys
    enc_rngkey, dec_rngkey = random.split(rngkey)
    
    ### 2.) encoder
    encoder = uniEncodeLSTM(hidden_size = hidden_dim,
                            n_layers = num_layers,
                            dropout_prob = dropout)
    
    fake_in = jnp.empty((batch_size, inseq_maxlen))
    encoder_initvars = encoder.init(rngs = enc_rngkey, 
                                    x_toenc = fake_in,
                                    training = False)
    
    ### 3.) decoder
    decoder = DecodeLSTM(hidden_size = hidden_dim,
                         n_layers = num_layers,
                         dropout_prob = dropout)
    
    fake_out = jnp.empty((batch_size, outseq_maxlen))
    decoder_initvars = decoder.init(rngs = dec_rngkey, 
                                    x_todec = fake_out,
                                    carry_todec = None,
                                    training = False)
    
    ### initialize train states
    encoder_trainstate = train_state.TrainState.create(apply_fn=encoder.apply, 
                                              params=encoder_initvars, 
                                              tx=tx)
    
    decoder_trainstate = train_state.TrainState.create(apply_fn=decoder.apply, 
                                              params=decoder_initvars, 
                                              tx=tx)
    
    return (encoder_trainstate, decoder_trainstate)



### bidirectional encoder, unidirectional encoder
def bidirec_init(hidden_dim, num_layers, dropout, batch_size, inseq_maxlen,
                 outseq_maxlen, tx, rngkey, merge_fn):
    ### 1.) rng keys
    enc_rngkey, dec_rngkey = random.split(rngkey)
    
    ### 2.) encoder
    encoder = biEncodeLSTM(hidden_size = hidden_dim,
                            n_layers = num_layers,
                            dropout_prob = dropout,
                            merge_fn = merge_fn)
    
    fake_in = jnp.empty((batch_size, inseq_maxlen))
    encoder_initvars = encoder.init(rngs = enc_rngkey, 
                                    x_toenc = fake_in,
                                    training = False)
    
    ### 3.) decoder
    decoder = DecodeLSTM(hidden_size = hidden_dim,
                         n_layers = num_layers,
                         dropout_prob = dropout)
    
    fake_out = jnp.empty((batch_size, outseq_maxlen))
    decoder_initvars = decoder.init(rngs = dec_rngkey, 
                                    x_todec = fake_out,
                                    carry_todec = None,
                                    training = False)
    
    ### initialize train states
    encoder_trainstate = train_state.TrainState.create(apply_fn=encoder.apply, 
                                              params=encoder_initvars, 
                                              tx=tx)
    
    decoder_trainstate = train_state.TrainState.create(apply_fn=decoder.apply, 
                                              params=decoder_initvars, 
                                              tx=tx)
    
    return (encoder_trainstate, decoder_trainstate)
    