#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 18:39:11 2023

@author: annabel

Train a seq2seq LSTM
"""
import os
import shutil

import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.training import train_state
import orbax.checkpoint
import optax

from Dataloader import *
from modelLayersUtils import _sum, _concatenate
from Initializers import *
from TrainTest import *


##################
### Parameters   #
##################
# loading data
training_filename = 'sine_trainy.tsv'
generation_filename = 'sine_testy.tsv'

# model params
hidden_dim = 64
num_layers = 1
dropout = 0

# extra options for specifically triggering bidirectional
bidirectional = True
merge_fn = _sum

# training loop params
num_epochs = 200
batch_size = 50
learning_rate = 0.001

# initial random key
rngkey = random.key(21)

# outfile prefixes
outfile_prefix = 'debug'


###########
### setup #
###########
# declare the default device to do everything on
jax.default_device = jax.devices("cpu")[0]


###############
### read data #
###############
training_dset = PairLoader(training_filename)
training_dl = DataLoader(training_dset, 
                         batch_size = batch_size, 
                         shuffle = True,
                         collate_fn = jax_collator)

generation_dset = PairLoader(generation_filename)
generation_dl = DataLoader(generation_dset, 
                           batch_size = batch_size, 
                           shuffle = False,
                           collate_fn = jax_collator)

(inseq_maxlen, outseq_maxlen) = training_dset.get_maxlens()


#########################
### initialize models   #
#########################
# this actually returns Train States, but if you needed it, could
# return the model objects too
tx = optax.adam(learning_rate)
rngkey, model_init_rngkey = random.split(rngkey, num=2)

if not bidirectional:
    enc_state, dec_state = unidirec_init(hidden_dim = hidden_dim, 
                                         num_layers = num_layers, 
                                         dropout = dropout, 
                                         batch_size = batch_size, 
                                         inseq_maxlen = inseq_maxlen,
                                         outseq_maxlen = outseq_maxlen, 
                                         tx = tx,
                                         rngkey = model_init_rngkey, 
                                         merge_fn=None)
else:
    enc_state, dec_state = bidirec_init(hidden_dim = hidden_dim, 
                                        num_layers = num_layers, 
                                        dropout = dropout, 
                                        batch_size = batch_size, 
                                        inseq_maxlen = inseq_maxlen,
                                        outseq_maxlen = outseq_maxlen, 
                                        tx = tx,
                                        rngkey = model_init_rngkey, 
                                        merge_fn= merge_fn)



#####################
### training loop   #
#####################
# save the checkpoints to a seprate directory, because I have no idea
# what intermediates are going to be output
enc_ckpt_dir = os.getcwd() + f'/{outfile_prefix}_ENCckpt'
dec_ckpt_dir = os.getcwd() + f'/{outfile_prefix}_DECckpt'

# for now, only save the best, but there's a lot of cool checkpointing
# options in orbax
options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=1)
enc_cman = checkpt_setup(output_dir = enc_ckpt_dir, 
                         orbax_options = options)
dec_cman = checkpt_setup(output_dir = dec_ckpt_dir, 
                         orbax_options = options)


# training loop time
best_epoch = -1
best_loss = 9999
rngkey, training_rngkey = random.split(rngkey, num=2)
for epoch in range(num_epochs):
    if epoch % 100 == 0:
        print(f'PROGRESS: {epoch}/{num_epochs}')
    
    # iterate through batch
    epoch_loss = 0
    for batch in training_dl:
        batch_loss, enc_state, dec_state = train_step(enc_state = enc_state, 
                                                      dec_state = dec_state, 
                                                      batch = batch, 
                                                      rngkey = training_rngkey,
                                                      epoch_num = epoch)
        epoch_loss = epoch_loss + batch_loss
    
    # get average loss
    ave_epoch_loss = epoch_loss/len(training_dl)
    
    # save the checkpoint based on ave_epoch_loss
    if ave_epoch_loss < best_loss:
        print(f'New best loss at epoch {epoch}: {ave_epoch_loss}')
        best_loss = ave_epoch_loss
        best_epoch = epoch
        
        # for some reason, need to explicitly include params and opt state
        enc_cman.save(step = epoch, items = {'model':enc_state, 
                                             'params': enc_state.params,
                                             'opt_state': enc_state.opt_state,
                                             'step': epoch})
        
        dec_cman.save(step = epoch, items = {'model':dec_state, 
                                             'params': dec_state.params,
                                             'opt_state': dec_state.opt_state,
                                             'step': epoch})

### clean up variables from training
del enc_state, dec_state


###########################
### generation/evaluation #
###########################
### load best model
# initialize some blank train states
rngkey, restoration_rngkey = random.split(rngkey, num=2)

if not bidirectional:
    best_enc_state, best_dec_state = unidirec_init(hidden_dim = hidden_dim, 
                                          num_layers = num_layers, 
                                          dropout = dropout, 
                                          batch_size = batch_size, 
                                          inseq_maxlen = inseq_maxlen,
                                          outseq_maxlen = outseq_maxlen, 
                                          tx = tx,
                                          rngkey = restoration_rngkey, 
                                          merge_fn=None)
else:
    best_enc_state, best_dec_state = bidirec_init(hidden_dim = hidden_dim, 
                                        num_layers = num_layers, 
                                        dropout = dropout, 
                                        batch_size = batch_size, 
                                        inseq_maxlen = inseq_maxlen,
                                        outseq_maxlen = outseq_maxlen, 
                                        tx = tx,
                                        rngkey = restoration_rngkey, 
                                        merge_fn= merge_fn)

# fill these in with the saved checkpoints
orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
best_enc_state = orbax_checkpointer.restore(f'{enc_ckpt_dir}/{best_epoch}/default', 
                                            item = best_enc_state)
best_dec_state = orbax_checkpointer.restore(f'{dec_ckpt_dir}/{best_epoch}/default', 
                                            item = best_dec_state)



### generate outputs from the validation set
pred_trajs = []
true_trajs = []
for batch in generation_dl:
    batch_out, true_outseqs = eval_batch(enc_state = best_enc_state, 
                                         dec_state = best_dec_state, 
                                         batch = batch)
    
    # update buckets holding all samples
    pred_trajs.append(batch_out)
    true_trajs.append(true_outseqs)

# concat all_traj along dimension 0
pred_trajs = jnp.concatenate(pred_trajs, axis=0)
true_trajs = jnp.concatenate(true_trajs, axis=0)


### check final validation loss
err = pred_trajs - true_trajs
square_err = jnp.square(err)
final_val_loss = jnp.mean(square_err)
print(f'Final loss on validation set: {final_val_loss}')



#%% (new cell in spyder IDE)
################################################
### use matplotlib to plot predicted vs true   #
### for a given sample trajectory              #
################################################
import matplotlib.pyplot as plt
sample_idx = -1


pred_samplevec = pred_trajs[sample_idx,:]
true_samplevec = true_trajs[sample_idx,:]
x_vec = range(len(pred_samplevec))

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(x_vec, pred_samplevec, label='pred', linestyle='-')
ax.plot(x_vec, true_samplevec, label='true', linestyle='-')
ax.legend()

