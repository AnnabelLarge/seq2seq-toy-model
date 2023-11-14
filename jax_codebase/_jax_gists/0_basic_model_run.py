#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 23:41:53 2023

@author: annabel

An example of instantiating a model, initializing layers, and passing
  an input in
"""
from flax import linen as nn
import jax
import jax.numpy as jnp
from jax import random


#########################
### 1.) BUILD A MODEL   #
#########################
class MLP(nn.Module):
    nfeats: int
    
    @nn.compact
    def __call__(self, x, carry):
        x = nn.Dense(features=self.nfeats, name='layer1')(x)
        carry = carry + 1
        return (x, carry)


###########################
### 2.) INITIALIZE MODEL  #
###########################
### instantiate model
mymodel = MLP(nfeats = 3)

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
