"""First use investigate dreams to save out a file with image loss for each desired agent,
then run this to plot them on the same graph."""

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


if __name__ == "__main__":
  basedir = f"{expanduser('~')}/logs/"

  to_compare  = [
    ['DRE-296', 'variables_train_agent_envindex0_001452500.pkl', 'freerange'],
    ['DRE-300', 'variables_train_agent_envindex0_001452500.pkl', '100k episodic'],
    ['DRE-292', 'variables_train_agent_envindex0_000500000.pkl', '1k episodic'],
  ]

  # to_compare  = [
  #   ['DRE-296', 'variables_train_agent_envindex0_000522500.pkl', '522500'],
  #   ['DRE-296', 'variables_train_agent_envindex0_001022500.pkl', '1022500'],
  #   ['DRE-296', 'variables_train_agent_envindex0_001522500.pkl', '1522500'],
  #   ['DRE-296', 'variables_train_agent_envindex0_002022500.pkl', '2022500'],
  #   # ['DRE-296', 'variables_train_agent_envindex0_002522500.pkl', '2522500'],
  # ]

  bufferid = 'GEN-EXAMPLE_EPS'
  which_ep = 1

  all_loss = []
  all_imgs = []
  all_intr_reward = []
  for exp in to_compare:
    expid = exp[0]
    checkpoint_name = exp[1]
    load_dir = f'{basedir}/{expid}/plots/{bufferid}/{checkpoint_name}'
    d = pickle.load(open(os.path.join(load_dir, 'img_loss.pkl'), 'rb'))

    all_loss.append(d['all_loss'][which_ep])
    all_imgs.append(d['all_imgs_w_burnin'][which_ep])
    all_intr_reward.append(d['all_intr_reward'][which_ep])

  burnin = d['burnin']
  imagine_for = d['imagine_for']

  # Plot img_loss
  legend_str = []
  for i in range(len(to_compare)):
    plt.plot(np.arange(-burnin, imagine_for), all_loss[i].flatten())
    legend_str.append(f'{to_compare[i][0], to_compare[i][2]}')
  plt.axvline(0, color='k', linestyle='--')
  plt.title(f'{bufferid}, ep {which_ep}')
  plt.ylabel('Image loss')
  plt.xlabel('Imagined steps')
  plt.legend(legend_str)
  plt.show()

  # Plot intr_reward
  # Plot img_loss
  legend_str = []
  for i in range(len(to_compare)):
    plt.plot(np.arange(-burnin, imagine_for), all_intr_reward[i].flatten())
    legend_str.append(f'{to_compare[i][0], to_compare[i][2]}')
  plt.axvline(0, color='k', linestyle='--')
  plt.title(f'{bufferid}, ep {which_ep}')
  plt.ylabel('Intrinsic reward')
  plt.xlabel('Imagined steps')
  plt.legend(legend_str)
  plt.show()


  # Plot img predictions
  plt.figure()
  for row in range(len(to_compare)):
    for col in range(len(all_imgs[row])):
      plt.subplot(len(to_compare), len(all_imgs[0]), row*len(all_imgs[0])+col+1)
      plt.imshow(all_imgs[row][col])
      plt.xticks([])
      plt.yticks([])
      if row == 0:
        plt.title(f"t={d['imgs_t'][col]}")
      if col == 0:
        plt.ylabel(f'{to_compare[row][0]}\n{to_compare[row][2]}')
  plt.tight_layout()
  plt.show()

  print('done')


