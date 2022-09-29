"""Overwrite the rewards in replay buffer"""
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
import time
import pickle

import elements
import common
from dreamerv2 import agent
import argparse
from tqdm import tqdm

from os.path import expanduser
import training.utils.loading_utils as lu
import training.utils.diagnosis_utils as du

tf.config.run_functions_eagerly(True)

if __name__ == "__main__":
  expid = 'GEN-2r' #'GEN-435'
  reward_type = 'yellow'

  replay_buffer_name = 'test_env_train_replay_1-1'
  new_replay_buffer = f'{replay_buffer_name}_new'

  home = expanduser("~")
  basedir = f"{home}/logs/{expid}"
  os.makedirs(f'{basedir}/{new_replay_buffer}', exist_ok=True)

  neps = len(os.listdir(f"{basedir}/{replay_buffer_name}/"))
  # neps = 100
  do_plot = True
  do_save_new_buffer = True

  TARGET_x = 6.5
  TARGET_y = 8.0 # -8.0 for other ball

  for which_ep in tqdm(range(1, neps)):
    nt = 501
    # eps = lu.load_eps(basedir, replay_buffer_name, batch_size, start_ind=which_batch*(batch_size+skip_size))

    ep, name, timestamp = lu.load_ep(basedir, which_ep=which_ep, buffer_name=replay_buffer_name)

    ep = dict(ep)
    r_orig = ep['reward']

    x = ep['absolute_position_agent0'][:, 0]
    y = ep['absolute_position_agent0'][:, 1]
    if reward_type == 'position':
      arena_size = 20
      reward = lambda x, y: (arena_size - np.sqrt((x-TARGET_x)**2 + (y-TARGET_y)**2))/arena_size
      r = reward(x, y)
    elif reward_type == 'yellow':
      ep['image']
      npixels = np.prod(ep['image'].shape[1:3])

      # blue
      # np.where((ep['image'][:, :, :, 0] < 10) & (ep['image'][:, :, :, 1] < 10) & (ep['image'][:, :, :, 2] > 200))[0]
      # yellow
      yellow = np.where((ep['image'][:, :, :, 0] > 190) &
                        (ep['image'][:, :, :, 1] > 190) &
                        (ep['image'][:, :, :, 2] < 10))[0]
      r = np.bincount(yellow)/(npixels/4)
      r = np.pad(r, (0, len(r_orig) - len(r)))
      r = r.astype(np.float32)
      # plt.figure(), plt.figure(), plt.imshow(ep['image'][np.argmax(r), :, :])
      # plt.title(f'{which_ep}:{np.argmax(r)}:{np.max(r):.5f}')
      # plt.show()
      print(len(r))

    if do_plot:
      plt.scatter(x, y,
                  # c=r,
                  c=r, alpha=np.clip(10*r, 0.0, 1.0),
                  # alpha=0.5,
                  s=0.2, cmap='coolwarm')
      # plt.clim([0, 1])
      plt.xlim([-20, 20])
      plt.ylim([-20, 20])
      plt.title(which_ep)
      plt.plot(TARGET_x, TARGET_y, 'k.')
      # plt.show()
    ep['reward'] = r
    if do_save_new_buffer:
      np.savez_compressed(f'{basedir}/{new_replay_buffer}/{name}', **ep)
  if do_plot:
    # plt.colorbar()
    plt.show()
  print('done')