#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 10:55:32 2023

@author: annabel
"""
import torch
from torch import nn, optim

def train_seq2seq(model, num_epochs, learning_rate, training_dl, model_file):
    """
    Train the model and save its best checkpoint
    
    TODO: I never implement anything that evaluates loss on test set, but this
          should be monitored through training too
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    best_loss = 99999
    for epoch in range(num_epochs):
        if epoch % 100 == 0:
            print(f'PROGRESS: {epoch}/{num_epochs}')
            
        # initialize loss storage for batch
        total_epoch_loss = 0
        
        # iterate through batches
        for batch in training_dl:
            # unpack
            inseqs, outseqs = batch
            
            # zero out the optimizer and compute loss with current params
            optimizer.zero_grad()
            batch_loss = model.compute_loss(source=inseqs,
                                            target=outseqs)
            
            # save the loss for this batch
            total_epoch_loss += batch_loss
            
            # backpropogate loss
            batch_loss.backward()
            optimizer.step()
            
            # clear vars
            del inseqs, outseqs, batch_loss
        
        # get an average epoch loss, decide whether or not to save model
        ave_epoch_loss = total_epoch_loss/len(training_dl)
        if ave_epoch_loss < best_loss:
            print(f'New best training loss at epoch {epoch}: {ave_epoch_loss}')
            torch.save(model.state_dict(), model_file)
            best_loss = ave_epoch_loss


def generate(best_model, test_dl):
    # loss function to use
    loss_fn = nn.MSELoss(reduction='mean')
    
    # place model in eval mode
    best_model = best_model.eval()
    
    pred_trajs = []
    true_trajs = []
    
    ### generate in a batched method (but I probably have the memory to
    ### do this in one go... use batching, for consistency with future code)
    for batch in test_dl:
        inseqs, true_outseqs = batch
        
        # generate predicted outseqs that are the same length as inseqs
        gen_length = inseqs.shape[1]
        
        # encode input sequences first
        enc_out, enc_hidden = best_model.encode(inseqs)
        
        # bucket variables to adjust through loop
        decoder_input = inseqs[:,-1].unsqueeze(1)
        decoder_hidden = enc_hidden
        batch_out = []
        
        # generate one token at a time
        for gen_idx in range(gen_length):
            # decode one time step
            step_out, decoder_hidden = best_model.decode(decoder_input, 
                                                         decoder_hidden)
            
            # append to batch_out
            batch_out.append(step_out)
            
            # the next step's input is the current step's output
            decoder_input = step_out
        
        # concat batch_out along dimension 1
        batch_out = torch.cat(batch_out, dim=1)
        
        # add to pred_trajs
        pred_trajs.append(batch_out)
        
        # update the true trajs
        true_trajs.append(true_outseqs)
    
    # concat all_traj along dimension 0
    pred_trajs = torch.cat(pred_trajs, dim=0)
    true_trajs = torch.cat(true_trajs, dim=0)
    
    # validation loss
    total_val_loss = loss_fn(pred_trajs, true_trajs)
    
    return (pred_trajs, true_trajs, total_val_loss)
