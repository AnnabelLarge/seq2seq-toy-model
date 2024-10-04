#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 10:27:15 2023

@author: annabel

"""
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from Dataloader import PairLoader
from Models import Seq2seqLSTMs
from TrainTest import train_seq2seq, generate


############
### params #
############
# loading data
training_filename = 'sine_trainy.tsv'
generation_filename = 'sine_testy.tsv'

# model params
hidden_size = 64
bidirection_enc = False
num_layers = 1
dropout = 0.001
model_file = 'lineardata_LSTMparams.pt'

# training loop params
batch_size = 50
num_epochs = 10
learning_rate = 0.001



###################################
### initialize dataloaders, model #
###################################
training_dset = PairLoader(training_filename)
training_dloader = DataLoader(training_dset, 
                              batch_size = batch_size, 
                              shuffle = False)

generation_dset = PairLoader(generation_filename)
generation_dloader = DataLoader(generation_dset, 
                                batch_size = batch_size, 
                                shuffle = False)

# input_size = training_dset.getInputSize()
input_size = 1

model = Seq2seqLSTMs(input_size= input_size, 
                   hidden_size= hidden_size, 
                   bidirection_enc= bidirection_enc, 
                   num_layers= num_layers, 
                   dropout = dropout)


#################
### train model #
#################
train_seq2seq(model = model, 
              num_epochs = num_epochs, 
              learning_rate = learning_rate,
              training_dl = training_dloader, 
              model_file = model_file)


###############################
### generate new trajectories #
###############################
# load the best model
del model
best_model = Seq2seqLSTMs(input_size= input_size, 
                       hidden_size= hidden_size, 
                       bidirection_enc= bidirection_enc, 
                       num_layers= num_layers, 
                       dropout = dropout)
best_model.load_state_dict(torch.load(model_file))

# generate
pred, true, generation_loss = generate(best_model = best_model, 
                                       test_dl = generation_dloader)

print(f'Final generation loss: {generation_loss}')


#%%
##############################################
### EXTRA: examine sample trajectories       #
###        compare against true trajectories #
##############################################
# could probably write a sticher that adds the correct xvectors... but don't
# actually spend that much more time on this pytorch example
import matplotlib.pyplot as plt
sample_idx = -10


pred_samplevec = pred[sample_idx,:].detach().numpy()
true_samplevec = true[sample_idx,:].detach().numpy()
x_vec = range(len(pred_samplevec))

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(x_vec, pred_samplevec, label='pred', linestyle='-')
ax.plot(x_vec, true_samplevec, label='true', linestyle='-')
