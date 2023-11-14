# seq2seq: A toy model

## Task description
The goal is a seq2seq model than can autoregressively generate a simple function output (like a simple time-series forecasting problem). Two toy datasets are given: 

1.  ```f(t) = 0.5t```
2.  ```f(t) = sin([2 * np.pi * 0.1] t)```

Training dataset generated from t=0 to t=239.75 (80% of data). Test dataset generated from t=240 to t=299.75 (20% of data).

## Model description

Our model uses the following: 
- LSTM encoder (either unidirectional and bidirectional can be implemented)
- unidirectional LSTM decoder

Loss function is mean squared error between predicted and true outputs.


## Pytorch codebase organization
- data
  - this folder contains the training and test sets; this is where the dataloader looks for things
- ```Dataloader.py```
  - this has custom dataloaders, to read time series data
- ```Models.py``` 
  - this stores the encoder and decoder
- ```TrainTest.py```
  - these functions are used for: 1.) training one epoch, and 2.) generation
- ```RUN.py```
  - this is where everything happens, including: 1.) loading data, 2.) instantiating a model, 3.) training the model using the training set, 4.) generating new trajectories prompted by the test set, 5.) plotting an example trajectory

## Jax codebase organization
- data
  - this folder contains the training and test sets; this is where the dataloader looks for things
- ```Dataloader.py```
  - this has custom dataloaders (using pytorch dataloaders, that output jax arrays)
- ```Initializers.py```
  - these functions separately initialize the encoder and decoder, returning two TrainState objects
- ```modelBlocks.py```
  - this stores the different types of encoders and decoders to choose from
  - uses layers defined separately (in ```modelUtils.py```)
- ```modelUtils.py```
  - this stores the individual LSTM layers (composed together in ```modelBlocks.py``` models)
  - this also has different concatenation functions, for combining the forward and reverse directions of the bidirectional LSTMs
- ```TrainTest.py```
  - these functions are used for: 1.) training one epoch, and 2.) generation
  - also has a helper utility for setting up orbax checkpoints
- ```RUN.py```
  - this is where everything happens, including: 1.) loading data, 2.) instantiating a model, 3.) training the model using the training set, 4.) generating new trajectories prompted by the test set, 5.) plotting an example trajectory
- ```_jax_gists```
  - this is some scratch work I did, to get the hang of instantiating jax models



## Todo
- integrate tensorboard
- in real workflows, I also monitor validation loss during training; implement this
- should turn the ```RUN.py``` scripts into jupyter notebooks
  - instead of plotting example trajectories, could stitch them all together (this would require annotating the original data with ```t```, or some clever code to work out order/windowing)
- the jax model is implemented as two separate models, but I think this could be united under one main model object with separate encoder/decoder methods (similar to the pytorch setup)


