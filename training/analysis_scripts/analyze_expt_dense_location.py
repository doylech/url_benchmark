"""Summarize logs across multiple runs with novel objects."""

import math

import PIL.Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
import scipy.signal
import seaborn as sns
from PIL import Image
import glob
import seaborn as sns

import training.analysis_scripts.analysis_utils as au

import matplotlib
matplotlib.use('module://backend_interagg')
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def simple_plot(ax):
  ax.spines.right.set_visible(False)
  ax.spines.top.set_visible(False)
  ax.yaxis.set_ticks_position('left')
  ax.xaxis.set_ticks_position('bottom')


def main():

  user = 'ikauvar'
  if user == 'ikauvar':
    user_info = {'local_user': 'saal2',
                 'gcloud_path':'/home/saal2/google-cloud-sdk/bin/gcloud',
                 'user': 'ikauvar',
                 'gcp_zone': 'us-central1-a'}
    plotdir = '/home/saal2/Dropbox/_gendreamer/plots/dense_location'
  elif user == 'cd':
    user_info = {'local_user': 'cd',
                 'gcloud_path':'/snap/bin/gcloud',
                 'user': 'cd',
                 'gcp_zone': 'us-central1-a'}

  ids = {  # freerange, reward
    # 'DRA-33': 't4-tf-1', # black
    # 'DRA-34': 't4-tf-2', # black
    # 'DRA-35': 't4-tf-3', # black
    # 'DRA-45': 't4-tf-3',  # Just testing sparse reward
    # 'DRA-37': 't4-tf-5',  # Dense goal
    # 'DRA-36': 't4-tf-4',  # Dense goal

    ## Good
    # 'DRA-47': 't4-tf-2',  # Dense goal
    # 'DRA-46': 't4-tf-1',  # Dense goal
    # 'DRA-57': 't4-tf-10', # Dense goal
    # 'DRA-56': 't4-tf-9', # Dense goal
    # 'DRA-38': 't4-tf-6',  # Dense goal

    ## Longer run
    'DRA-122': 't4-tf-9',  # Dense goal
    'DRA-123': 't4-tf-10',  # Dense goal
    'DRA-124': 't4-tf-11',  # Dense goal
    'DRA-125': 't4-tf-12',  # Dense goal
  }

  labels = {
    'ball3': 'familiar',
    'ball4': 'novel',
  }
  colors = {
    'ball3': 'orange',
    'ball4': 'magenta',
  }
  ball_ids = list(colors.keys())


  force_download = 0

  env_nums = [0]

  saverate = 20
  # xlim = int(3000e3 / saverate)
  xlim = int(4e6 / saverate)

  do_plot_positions = True
  do_plot_distance_from_target = True
  do_plot_reward = True

  # TO PLOT:
    # Distance from target over time
    # Reward over time

  all_df = None
  all_dfm = None
  changepoints = []
  for env_num in env_nums:
    for id, remote in ids.items():
      df = au.load_log(id, env_num, remote, user_info, force_download=force_download)
      dfm = au.load_metrics(id, env_num, remote, user_info, force_download=force_download)

      save_freq = np.diff(df['total_step'].to_numpy())[1]

      if all_df is None:
        all_df = df
        all_dfm = dfm
      else:
        all_df = pd.concat((all_df, df))
        all_dfm = pd.concat((all_dfm, dfm))

      # Plot reward
    if do_plot_reward:
      all_reward = []
      all_step = []
      for id in all_dfm['id'].unique():
        plot_dfm = all_dfm[all_dfm['id'] == id]

        plot_dfm = plot_dfm[['step', 'ee/env-0/expl_reward_mean']].dropna()
        reward = plot_dfm['ee/env-0/expl_reward_mean']
        step = plot_dfm['step']
        all_reward.append(reward.to_numpy())
        all_step.append(step.to_numpy())

      min_t = min([len(x) for x in all_reward])
      all_reward = [x[:min_t] for x in all_reward]
      all_step = [x[:min_t] for x in all_step]
      all_reward = np.vstack(all_reward)
      tt = np.vstack(all_step)

      rew_m = np.median(all_reward, axis=0)
      rew_s = scipy.stats.sem(all_reward, axis=0)
      # tt = saverate*np.arange(len(rew_m))
      plt.figure(figsize=(3,3))
      plt.plot(tt.T, all_reward.T)
      plt.ylabel('Reward')
      plt.xlabel('Steps')
      plt.ylim([0, 20])
      simple_plot(plt.gca())
      plt.xlim([0, xlim*saverate])
      plt.tight_layout()
      plt.title(str(list(ids.keys())))
      plt.savefig(os.path.join(plotdir, 'reward_indiv.pdf'), transparent=True)
      plt.savefig(os.path.join(plotdir, 'reward_indiv.svg'), transparent=True)
      plt.savefig(os.path.join(plotdir, 'reward_indiv.png'))

      # Plot mean

      plt.figure(figsize=(3,3))
      plt.fill_between(tt[0, :], rew_m - rew_s, rew_m + rew_s, alpha=0.4)
      plt.plot(tt[0, :], rew_m)
      plt.ylabel('Reward')
      plt.xlabel('Steps')
      simple_plot(plt.gca())
      plt.ylim([0, 20])
      plt.xlim([0, xlim*saverate])

      plt.tight_layout()
      plt.title(str(list(ids.keys())))
      plt.savefig(os.path.join(plotdir, 'reward.pdf'), transparent=True)
      plt.savefig(os.path.join(plotdir, 'reward.svg'), transparent=True)
      plt.savefig(os.path.join(plotdir, 'reward.png'))

    if do_plot_positions:
      for id in all_df['id'].unique():
        plot_df = all_df[all_df['id'] == id]

        plt.figure(figsize=(3, 3))
        plt.plot(plot_df['agent0_xloc'][:xlim], plot_df['agent0_yloc'][:xlim], 'k.',
                 markersize=1, alpha=0.2)
        plt.scatter(plot_df['target_xloc'][0], plot_df['target_yloc'][0], s=400, facecolors='none', edgecolors='r')
        for b in ball_ids:
          try:
            plt.plot(plot_df[f'{b}_xloc'][:xlim], plot_df[f'{b}_yloc'][:xlim], '.',
                     markersize=2, color=colors[b], alpha=0.5)
          except: pass
        plt.title(f'Position {id}')
        plt.show()

    if do_plot_distance_from_target:
      all_dist = []
      for id in all_df['id'].unique():
        plot_df = all_df[all_df['id'] == id]
        print(len( plot_df['agent0_xloc']))
        x = plot_df['agent0_xloc'][:xlim]
        y = plot_df['agent0_yloc'][:xlim]
        target_x = plot_df['target_xloc'][0]
        target_y = plot_df['target_yloc'][0]

        dist = np.sqrt((x - target_x)**2 + (y - target_y)**2)
        all_dist.append(dist.to_numpy())
        # plt.plot(dist)
        # plt.title(id)
        # plt.show()

      all_dist = np.vstack(all_dist)
      dist_m = np.median(all_dist, axis=0)
      dist_s = scipy.stats.sem(all_dist, axis=0)
      tt = saverate*np.arange(len(dist_m))

      all_dist = all_dist[:, ::10]
      dist_m = dist_m[::10]
      dist_s = dist_s[::10]
      tt = tt[::10]

      plt.figure(figsize=(3,3))
      plt.plot(np.tile(tt, [all_dist.shape[0], 1]).T, all_dist.T)
      plt.ylabel('Distance from target')
      plt.xlabel('Steps')
      simple_plot(plt.gca())
      plt.tight_layout()

      # Plot mean
      plt.figure(figsize=(3,3))
      plt.fill_between(tt, dist_m - dist_s, dist_m + dist_s, alpha=0.4)
      plt.plot(tt, dist_m)
      plt.ylabel('Distance from target')
      plt.xlabel('Steps')
      simple_plot(plt.gca())
      plt.ylim([0, 25])
      plt.tight_layout()
      plt.title(str(list(ids.keys())))
      plt.savefig(os.path.join(plotdir, 'distance_from_target.pdf'), transparent=True)
      plt.savefig(os.path.join(plotdir, 'distance_from_target.svg'), transparent=True)
      plt.savefig(os.path.join(plotdir, 'distance_from_target.png'))
      plt.show()




if __name__ == "__main__":
  main()