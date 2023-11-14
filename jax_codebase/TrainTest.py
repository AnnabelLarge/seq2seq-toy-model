#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 23:02:29 2023

@author: annabel_large


"""
import jax
import jax.numpy as jnp
from jax import random
import orbax.checkpoint
import os
import shutil


@jax.jit
def train_step(enc_state, dec_state, batch, rngkey, epoch_num):
    # unpack
    inseqs, true_outseqs = batch
    
    # create the inputs for teacher forcing, by concatenating the last column
    #   inseqs with all but last column of true_outseqs
    # x_to_cat = jnp.expand_dims(x[:, -1], axis=1)
    # y_to_cat = y[:, :-1]
    # out = jnp.concatenate([x_to_cat, y_to_cat], axis=-1)
    teachforc_inputs = jnp.concatenate([jnp.expand_dims(inseqs[:, -1], axis=1),
                                        true_outseqs[:, :-1]],
                                       axis = 1)
    
    # fold in the epoch number to the RNG key, to produce a new random key
    lstm_key = random.fold_in(rngkey, epoch_num)
    lstm_key, enc_dropout_key, dec_dropout_key = random.split(lstm_key, num=3)
    
    # define the loss function
    def evaluate_MSELoss(enc_ps, dec_ps):
        # apply the encoder and get the hidden state
        enc_carry, _ = enc_state.apply_fn(variables = enc_ps,
                                         x_toenc = inseqs,
                                         training=True, 
                                         rngs={'dropout': enc_dropout_key})
        
        # apply the decoder with teacher forcing, and get the final output
        _, pred_outseq = dec_state.apply_fn(variables = dec_ps,
                                            x_todec = teachforc_inputs,
                                            carry_todec = enc_carry,
                                            training=True,
                                            rngs= {'dropout': dec_dropout_key})
        
        # evaluate MSE loss
        err = pred_outseq - true_outseqs
        square_err = jnp.square(err)
        loss = jnp.mean(square_err)
        
        return loss, pred_outseq
    
    # has_aux: Indicates whether fun returns a pair where the first element is 
    # considered the output of the mathematical function to be differentiated 
    # and the second element is auxiliary data
    enc_grad_fn = jax.value_and_grad(evaluate_MSELoss, argnums=0, has_aux=True)
    dec_grad_fn = jax.value_and_grad(evaluate_MSELoss, argnums=1, has_aux=True)
    
    # apply gradients to both
    # weirdly, can't pass enc_ps = enc_state.params or 
    # dec_ps = dec_state.params without jax getting angry at me
    _, enc_grads = enc_grad_fn(enc_state.params, dec_state.params)
    (batch_loss, _), dec_grads = dec_grad_fn(enc_state.params, dec_state.params)
    
    # update both train states with same gradient
    enc_state = enc_state.apply_gradients(grads=enc_grads)
    dec_state = dec_state.apply_gradients(grads=dec_grads)
    
    return (batch_loss, enc_state, dec_state)
    

@jax.jit
def eval_batch(enc_state, dec_state, batch):
    # unpack the batch; only allow generation up to length of inseqs
    inseqs, true_outseqs = batch
    gen_length = inseqs.shape[-1]
    
    
    # encode the input; don't need rng key for inference
    enc_carry, _ = enc_state.apply_fn(variables = enc_state.params,
                                          x_toenc = inseqs,
                                          training=False)
    
    # bucket variables to adjust through loop
    decoder_input = jnp.expand_dims(inseqs[:,-1], axis=1)
    decoder_carry = enc_carry
    batch_out = []
    
    # start generation
    for gen_idx in range(gen_length):
        # # decode one time step
        # step_out, decoder_hidden = best_model.decode(decoder_input, 
        #                                              decoder_hidden)
        decoder_carry, step_out = dec_state.apply_fn(variables = dec_state.params,
                                            x_todec = decoder_input,
                                            carry_todec = decoder_carry,
                                            training=False)
        
        # append to batch_out
        batch_out.append(step_out)
        
        # the next step's input is the current step's output
        decoder_input = step_out
    
    
    # concatenate the output
    batch_out = jnp.concatenate(batch_out, axis=-1)
    
    return (batch_out, true_outseqs)




### use this to wrap up the checkpoint setup
def checkpt_setup(output_dir, orbax_options):
    # see if output directory exists; if it does, remove it
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # setup up checkpointer
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_manager = orbax.checkpoint.CheckpointManager(directory=output_dir, 
                                            checkpointers = orbax_checkpointer,
                                            options = orbax_options)
    
    return checkpoint_manager



    
