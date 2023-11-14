#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 18:48:28 2022

@author: annabel_large

use the pytorch dataloader to get a batch, then output to jax array

used the solution from here:
    https://jax.readthedocs.io/en/latest/notebooks/\
    Neural_Network_and_Data_Loading.html

"""
import torch
from torch.utils.data import Dataset, DataLoader,default_collate
import numpy as np
from jax import numpy as jnp
from jax.tree_util import tree_map

def jax_collator(batch):
    return tree_map(jnp.asarray, default_collate(batch))


class PairLoader(Dataset):
    def __init__(self, filename):
        # read the matrix of labels, y_mat
        print(f'Reading data from {filename}')
        y_mat = np.genfromtxt(f'./data/{filename}', delimiter='\t')
        
        # divide the vectors in half; let first half be the feature, 
        # and the second half be the target
        vec_len = y_mat.shape[1]
        split_idx = int(vec_len/2)
        self.inseq = y_mat[:, 0:split_idx]
        self.outseq = y_mat[:, split_idx:]
        
    def __len__(self):
        return jnp.asarray(self.inseq.shape[0])
    
    def __getitem__(self, idx):
        sample_inseq = self.inseq[idx, :]
        sample_outseq = self.outseq[idx, :]
        return (sample_inseq, sample_outseq)
    
    def getJaxItem(self, idx):
        """
        retrieve one sample, but as a jax array
        """
        sample_inseq, sample_outseq = self.__getitem__(idx)
        return (jnp.array(sample_inseq), jnp.array(sample_outseq))
    
    def get_maxlens(self):
        """
        return the maximum sequence length of the input and output sequences
        """
        max_inseq_length = self.inseq.shape[1]
        max_outseq_length = self.outseq.shape[1]
        return (max_inseq_length, max_outseq_length)
        


#%%
#########################
### TEST FUNCTIONS HERE #
#########################
if __name__ == '__main__':
    ### get a sample
    y = PairLoader(filename='linear_trainy.tsv')
    ex_y = y.getJaxItem(0)
    
    
    ### get a batch
    batch_size = 6
    num_feats = 18
    
    y_dl = DataLoader(y, 
                      batch_size = batch_size, 
                      shuffle = True,
                      collate_fn = jax_collator)
    ex_batch = list(y_dl)[-1]
