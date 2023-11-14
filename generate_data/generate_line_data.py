#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 16:13:19 2023

@author: annabel

Generate series training data by isolating different fixed-width subarrays 
  from a line

"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt


def generate_data(t_min, t_max, t_incr, slope, intercept, sample_len):
    """
    formula of a line: f(t) = slope * t + intercept
    
    Input these parameters, output a series of numbers
    
    Also plot the function, for visualization
    """
    ### make t_vector
    t_vec = np.arange(t_min, t_max+t_incr, t_incr)
    
    
    ### make y_vector
    y_vec = (slope * t_vec) + intercept
    
    
    ### plot the complete dataset, for easy visualization
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(t_vec, y_vec, '.-')
    mytit = (f'Line from {min(t_vec)} to {max(t_vec)}\n'+
             f'({slope} * t) + {intercept}')
    ax.set_title(mytit, size=14)
    ax.grid()
    
    
    ### make fixed-width slices into y, to make the time-series data
    out_seqs_y = sliding_window_view(x = y_vec, window_shape = sample_len)
    
    return out_seqs_y



if __name__ == '__main__':
    ### inputs for generator
    # parameters of the line
    slope = 0.5
    intercept = 0
    
    # the time points to generate over
    t_min = 0
    t_max = 1
    t_incr = 0.001
    
    # how long should the samples be
    # each sample is concat(infeats, outfeats)
    # that is, given the first infeats of the time series, predict the next
    #   outfeats of the time series
    infeat_len = 18
    outfeat_len = 18
    sample_len = infeat_len + outfeat_len
    
    
    ### generate numpy matrices
    y_mat = generate_data(t_min, t_max, t_incr, slope, intercept, sample_len)
    
    
    ### split into train and test sets
    # first 80% is training; last 20% is testing
    split_idx = int(0.80 * len(y_mat))
    train_y = y_mat[0:split_idx, :]
    test_y = y_mat[split_idx:, :]
    
    
    ### output to flat text files
    # will have to load from text files, when doing the real seq2seq task
    np.savetxt('train_y.tsv', train_y, delimiter='\t')
    np.savetxt('test_y.tsv', test_y, delimiter='\t')
    
    


