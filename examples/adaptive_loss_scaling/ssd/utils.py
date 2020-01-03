"""
Utility functions for analysing the results.
"""

import os
import numpy as np
import pandas as pd

def plot_loss_scale_dist(train_dir, ax):
  """ Plot the distribution of loss scales across all layers.
  
  The log file is in CSV format. Example rows:

    ,iter,label,key,val
    0,0,AdaLossConvolution2DFunction,unbound,1.0
    1,0,AdaLossConvolution2DFunction,bound,1.0
    2,0,AdaLossConvolution2DFunction,power_of_two,1.0
    3,0,AdaLossConvolution2DFunction,final,1.0
    4,0,AdaLossConvolution2DFunction,unbound,1.0
    5,0,AdaLossConvolution2DFunction,bound,1.0
    6,0,AdaLossConvolution2DFunction,power_of_two,1.0
    7,0,AdaLossConvolution2DFunction,final,1.0
    8,0,AdaLossConvolution2DFunction,unbound,1.0965939

  Explanation:
  - "iter" refers to the iteration ID.
  - "key" stands for the stage name of the loss scaling calculation, including "unbound", "bound", "power_of_two". 
  - "label" is the function ID.
  - "val" is the actual loss scale value.


  Args:
    train_dir(str): path to where log files are stored.
      There should exist a file called "loss_scale.csv"
  """
  log_path = os.path.join(train_dir, 'loss_scale.csv')
  if not os.path.isfile(log_path):
    raise FileNotFoundError("'loss_scale.csv' does not exist in train_dir: {}".format(train_dir))

  # load data
  df = pd.read_csv(log_path, index_col=0)
  
  # extract iteration IDs
  iters = np.unique(df['iter'].values)
  keys = np.unique(df['key'].values)

  # calculate how many layers logged in one iteration.
  N = len(df[(df['iter'] == iters[0])].index) // len(keys)

  # all loss scales
  # we only take care of the case when key equals to 'final'
  arr = df[df['key'] == 'final']['val'].values
  arr = arr.reshape([len(iters), N])
  arr = arr[:, ::-1] # reverse layer ID

  # calculate mean of loss scales for each layer
  mean = np.mean(arr, axis=0) # mean accross iterations 
  std = np.std(arr, axis=0)

  # plot to ax
  ax.bar(np.arange(len(mean)), mean, yerr=std)
