"""
Load episodes from replay buffer.
Quantify features of the episodes (such as how many magenta pixels).

See download_ckpt.sh to get necessary files from remote server for running this locally.
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

tf.config.run_functions_eagerly(True)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Specify episode arguments.')
  parser.add_argument('--buffer_name', default='train_replay_0')
  parser.add_argument('--expid',    default='DRA-159')
  args = parser.parse_args()

  expid = args.expid # For Fig 1: 'DRE-354'   # 'DRE-377'
  buffer_name = args.buffer_name
  home = expanduser("~")
  basedir = f"{home}/logs/{expid}"
  buffer_basedir = f"{basedir}/{buffer_name}"
  plotdir = f"{basedir}/plots"
  os.makedirs(plotdir, exist_ok=True)

  npx_magenta = []
  npx_yellow = []

  neps = len(os.listdir(buffer_basedir))
  for which_ep in range(neps):
    # Load episodes
    ep, traj_name, traj_timestamp = lu.load_ep(basedir, which_ep,
                                               buffer_name=buffer_name)
    imgs = ep['image']

    colors = {'yellow':  {'color_max': [255, 255, 20], 'color_min': [160, 160, 0]},
              'magenta': {'color_max': [255, 20, 255], 'color_min': [75, 0, 75]}}
    yellow = du.amount_of_color(imgs, colors['yellow']['color_min'], colors['yellow']['color_max']).numpy()
    magenta = du.amount_of_color(imgs, colors['magenta']['color_min'], colors['magenta']['color_max']).numpy()

    npx_magenta.append(magenta)
    npx_yellow.append(yellow)

  npx_yellow = np.hstack(npx_yellow).T
  npx_magenta = np.hstack(npx_magenta).T

  tt = np.arange(len(npx_yellow))*2
  plt.plot(tt, npx_yellow, 'orange')
  plt.plot(tt, npx_magenta, 'magenta')
  plt.xlabel('Steps')
  plt.ylabel('Fraction of pixels')
  plt.title(f'{expid}: {buffer_name}, yellow & magenta pixels')
  plt.savefig(os.path.join(plotdir, f'pixel_frac_{buffer_name}.png'))
  plt.show()

  tt = np.arange(len(npx_yellow))*2
  plt.plot(tt, np.cumsum(npx_yellow), 'orange')
  plt.plot(tt, np.cumsum(npx_magenta), 'magenta')
  plt.xlabel('Steps')
  plt.ylabel('Cumulative fraction of pixels')
  plt.title(f'{expid}: {buffer_name}, yellow & magenta pixels')
  plt.savefig(os.path.join(plotdir, f'pixel_frac_cum_{buffer_name}.png'))
  plt.show()

  print('done')
