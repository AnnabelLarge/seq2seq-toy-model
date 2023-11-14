#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 00:34:55 2023

@author: annabel

learning how to organize from:
  https://github.com/google/flax/blob/main/examples/nlp_seq

Crucial lesson from this: only need to initialize model parameters from the TOP
  MODEL LAYER (the intermediate blocks will get taken care of automatically)

TODO: how to use different kernel init and bias init
"""
from flax import linen as nn
import jax
import jax.numpy as jnp
from jax import random


#########################
### 1.) BUILD A MODEL   #
#########################
### define custom block
class MLPBlock(nn.Module):
    nfeats: int
    layername: str
    
    @nn.compact
    def __call__(self, x, carry):
        x = nn.Dense(features=self.nfeats, name=f'{self.layername}')(x)
        carry = carry + 1
        return (x, carry)


### define model, composed of user-defined number of blocks
class MLPLayers(nn.Module):
    nlayers: int
    nfeats: int
    
    @nn.compact
    def __call__(self, x, carry):
        for i in range(self.nlayers):
            x, carry = MLPBlock(nfeats= self.nfeats,
                                layername= f'block{i}')(x = x, carry = carry)
        
        return (x, carry)


###########################
### 2.) INITIALIZE MODEL  #
###########################
### instantiate model
mymodel = MLPLayers(nlayers=2,
                    nfeats=3)

### initialize variables
# wrap this in a function to hit it with that jitterbug
@jax.jit
def initialize_variables(init_rng):
    fake_x = jnp.empty((1, mymodel.nfeats))
    fake_carry = jnp.empty((1,6))
    
    init_variables = mymodel.init(rngs = init_rng, 
                                    x = fake_x,
                                    carry = fake_carry)
    return init_variables

init_variables = initialize_variables(init_rng = random.key(0))


#####################
### 3.) RUN MODEL   #
#####################
real_x = jnp.array([47,51,2])
real_carry = jnp.array([1,2,3,4,5,6])
y = mymodel.apply(variables=init_variables,
                  x = real_x,
                  carry = real_carry)
