#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 18:48:28 2022

@author: annabel_large

"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class PairLoader(Dataset):
    def __init__(self, filename):
        # read the matrix of labels, y_mat
        print(f'Reading data from {filename}')
        y_mat = np.genfromtxt(f'./data/{filename}', delimiter='\t', dtype=np.float32)
        if len(y_mat.shape) == 1:
            y_mat = np.expand_dims(y_mat, axis=0)
        
        # divide the vectors in half; let first half be the feature, 
        # and the second half be the target
        vec_len = y_mat.shape[1]
        split_idx = int(vec_len/2)
        self.inseq = y_mat[:, 0:split_idx]
        self.outseq = y_mat[:, split_idx:]
        
    def __len__(self):
        return self.inseq.shape[0]
    
    def __getitem__(self, idx):
        sample_inseq = self.inseq[idx, :]
        sample_outseq = self.outseq[idx, :]
        return (sample_inseq, sample_outseq)
    
    def getInputSize(self):
        return self.inseq.shape[1]
    
    def getJaxItem(self, idx):
        pass
        


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
                      shuffle = True)
    ex_batch = list(y_dl)[-1]
