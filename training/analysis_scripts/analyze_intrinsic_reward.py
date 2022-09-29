"""Load computed intrinsic reward and make plots,
use after diagnose_intrinsic_reward.py
"""
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

# from training.analysis_scripts.diagnose_intrinsic_reward import plot_curious_frames

tf.config.run_functions_eagerly(True)

if __name__ == "__main__":
  expid = 'p2e_fiddle_dreamer-launch'
  info = {
    'expid': expid,
    'replay_buffer_name': 'train_replay_0',
    'checkpoint_name': 'variables_pretrained_env0.pkl',
    'batch_size': 2,
    'skip_size': 12, #4,
    'n_batches': 21, #64,
    'nt': 501,
    'plot_lims': [-20,20],
    'basedir': f"{expanduser('~')}/logs/{expid}"
  }

  plot_dir = f'{info["basedir"]}/plots2'
  os.makedirs(plot_dir, exist_ok=True)

  fname = f'{info["basedir"]}/intr_rew_{info["replay_buffer_name"]}_nt{info["nt"]}__' \
          f'{info["checkpoint_name"]}__{info["batch_size"]}_{info["n_batches"]}_{info["skip_size"]}.pkl'
  with open(fname, 'rb') as f:
    dd = pickle.load(f)

  flat_xyz = dd['all_xyz_v'].reshape(np.prod(dd['all_xyz_v'].shape[:2]), -1)
  flat_image_likes = dd['all_image_likes_v'].reshape(np.prod(dd['all_image_likes_v'].shape[:2]), -1)
  flat_reward_likes = dd['all_reward_likes_v'].reshape(np.prod(dd['all_reward_likes_v'].shape[:2]), -1)
  flat_kl = dd['all_kl_v'].reshape(np.prod(dd['all_kl_v'].shape[:2]), -1)
  discounts = dd['discounts']
  flat_ep_rewards = {}
  for d in discounts:
    flat_ep_rewards[d] = dd['all_ep_rewards_v'][d].reshape(np.prod(dd['all_ep_rewards_v'][d].shape[:2]), -1)


  du.make_plots(flat_xyz, flat_image_likes, flat_reward_likes, flat_kl, flat_ep_rewards,
             discounts, info, plot_dir, dd['all_ep_rewards_v'])


  print(ddd.keys())