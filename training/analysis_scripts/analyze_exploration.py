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

def main():

  user = 'ikauvar'
  if user == 'ikauvar':
    user_info = {'local_user': 'saal2',
                 'gcloud_path':'/home/saal2/google-cloud-sdk/bin/gcloud',
                 'user': 'ikauvar',
                 'gcp_zone': 'us-central1-a'}
  elif user == 'cd':
    user_info = {'local_user': 'cd',
                 'gcloud_path':'/snap/bin/gcloud',
                 'user': 'cd',
                 'gcp_zone': 'us-central1-a'}

  # ids = {  # freerange, color
  #   'DRE-355': 't4-tf-4', # color
  #   'DRE-356': 't4-tf-5', # color
  # }
  ids = {  # freerange, black
    'DRE-354': 't4-tf-2', # black
    'DRE-346': 't4-tf-8', # black
    'DRE-348': 't4-tf-9', # black
    'DRE-475': 't4-tf-9',
  }
  # ids = {  # random policy playground
  #   'DRE-376': 't4-tf-2',
  #   'DRE-375': 't4-tf-1',
  #   'DRE-378': 't4-tf-2',
  #   'DRE-377': 't4-tf-1',
  # }

  ids = {  # freerange, reward
    # 'DRA-33': 't4-tf-1', # black
    # 'DRA-34': 't4-tf-2', # black
    # 'DRA-35': 't4-tf-3', # black
    # 'DRA-45': 't4-tf-3',  # Just testing sparse reward
    # 'DRA-38': 't4-tf-6',  # Dense reward
    # 'DRA-37': 't4-tf-5',  # Dense reward
    # 'DRA-36': 't4-tf-4',  # Dense reward
    'DRA-47': 't4-tf-2',  # Dense reward
    'DRA-46': 't4-tf-1',  # Dense reward
  }

  labels = {
    'ball3': 'familiar',
    'ball4': 'novel',
  }
  colors = {
    'ball3': 'orange',
    'ball4': 'magenta',
  }


  force_download = 0

  env_nums = [0]

  saverate = 20
  # xlim = int(100e3 / saverate)
  xlim = int(3000e3 / saverate)

  # xlim = np.arange(0.5e6, 5.5e6, 0.5e6)/saverate
  # xlim = np.array([0.5e6, 1e6, 1.5e6, 2e6, 2.5e6, 3e6])/saverate
  # xlim = np.array([2e6, 5e6, 8e6, 11e6])/saverate
  # xlim = xlim.astype(int)
  do_plot_trajs = False

  ball_ids = list(colors.keys())

  all_df = None
  changepoints = []
  for env_num in env_nums:
    for id, remote in ids.items():
      df = au.load_log(id, env_num, remote, user_info, force_download=force_download)
      save_freq = np.diff(df['total_step'].to_numpy())[1]

      if all_df is None:
        all_df = df
      else:
        all_df = pd.concat((all_df, df))

    # nf = save_freq*1e3*np.array([0, 5, 10, 15, 20, 25])
    nf = np.array([0, 3e4, 6e4])

    if do_plot_trajs:
      if type(xlim) == int:
        xlim_list = [xlim]
      else:
        xlim_list = xlim
      for id in all_df['id'].unique():
        for xl in xlim_list:
          plot_df = all_df[all_df['id'] == id]

          plt.figure(figsize=(3, 3))
          plt.plot(plot_df['agent0_xloc'][:xl], plot_df['agent0_yloc'][:xl], 'k.',
                   markersize=1, alpha=0.2)
          plt.xlim(-30, 30)
          plt.ylim(-30, 30)
          plt.axis('off')
          plt.title(f'{id}: {xl*saverate/1e6} M')
        plt.show()

    for id in all_df['id'].unique():
      plot_df = all_df[all_df['id'] == id]

      plt.figure(figsize=(8, 3))
      plt.subplot(1,2,1)
      plt.plot(plot_df['agent0_xloc'][:xlim], plot_df['agent0_yloc'][:xlim], 'k.',
               markersize=1, alpha=0.2)
      plt.scatter(plot_df['target_xloc'][0], plot_df['target_yloc'][0], s=400, facecolors='none', edgecolors='r')
      for b in ball_ids:
        try:
          plt.plot(plot_df[f'{b}_xloc'][:xlim], plot_df[f'{b}_yloc'][:xlim], '.',
                   markersize=2, color=colors[b], alpha=0.5)
        except: pass
      plt.title(f'Position {id}')

      plt.subplot(1,2,2)
      legend_str = []
      for b in ball_ids:
        try:
          # plt.plot(scipy.signal.savgol_filter(df[f'collisions_ball{b}/shell'], 1001, 3))
          tt = saverate*np.arange(np.cumsum(plot_df[f'collisions_{b}/shell'])[:xlim].shape[0])
          plt.plot(tt, np.cumsum(plot_df[f'collisions_{b}/shell'])[:xlim],
                   color=colors[b])
          legend_str.append(f'{b}:' + labels[f'{b}'])
          plt.xlabel('Steps in env')
        except: pass
      # [plt.axvline(x/saverate, linestyle='--', color='k') for x in nf[1:]]
      [plt.axvline(x, linestyle='--', color='k') for x in nf[1:]]
      plt.legend(legend_str, bbox_to_anchor=[1.05, 1])
      plt.title(f'Cumulative collisions: env {env_num}')
      plt.tight_layout()
      plt.show()


    do_summary = False
    if do_summary:
      for ii in range(len(nf)-1):
        nf1 = nf[ii]
        nf2 = nf[ii+1]
        env_df = all_df[all_df['env_num']==env_num]
        plot_df = env_df[(env_df['total_step'] > nf1) & (env_df['total_step'] < nf2)]
        plt.figure()
        all_means = []
        all_ball_means = []
        all_ball_x = []
        for ind, b in enumerate(ball_ids):
          ball_means = []
          for id in plot_df['id'].unique():
            vals = plot_df[plot_df['id']==id][f'collisions_{b}/shell']
            ball_means.append(np.mean(vals.to_numpy()))
          all_means.append(ball_means)
          all_ball_means.append(ball_means)
          all_ball_x.append([ind]*len(ball_means))
          plt.plot([ind]*len(ball_means), ball_means, '.', color=colors[b])
          plt.bar(ind, np.mean(ball_means), color=colors[b], alpha=0.2)
        plt.plot(np.array(all_ball_x), np.array(all_ball_means), 'k')
        plt.xticks(np.arange(len(ball_ids)), [labels[f'{b}'] for b in ball_ids])
        plt.title(f'Mean collisions in {nf1:.0e} to {nf2:.0e} timesteps')
        plt.suptitle(f'Env{env_num}, {all_df["id"].unique()}')
        plt.show()

    # Plot all cumsum combined across expts
    plt.figure()
    env_df = all_df[all_df['env_num'] == env_num]
    plot_df = env_df
    for id in plot_df['id'].unique():
      legend_str = []
      for b in ball_ids:
        try:
          # plt.plot(scipy.signal.savgol_filter(df[f'collisions_ball{b}/shell'], 1001, 3))
          tt = saverate*np.arange(len(plot_df[plot_df['id']==id][f'collisions_{b}/shell']))
          plt.plot(tt, np.cumsum(plot_df[plot_df['id']==id][f'collisions_{b}/shell']),
                   color=colors[b])
          legend_str.append(f'{b}:' + labels[f'{b}'])
          plt.xlabel('Steps in env')

        except: pass
    plt.legend(legend_str, bbox_to_anchor=[1.05, 1])
    plt.title(f'Cumulative collisions: env {env_num}')
    plt.suptitle(f'Env{env_num}, {all_df["id"].unique()}')
    plt.tight_layout()
    plt.show()

    plt.figure()
    env_df = all_df[all_df['env_num'] == env_num]
    plot_df = env_df
    all_avg_collisions = {}
    for b in ball_ids:
      all_avg_collisions[b] = []
    for id in plot_df['id'].unique():
      legend_str = []
      for b in ball_ids:
        try:
          w_k = 100
          N = int(w_k*1e3/saverate)
          avg_collisions = np.convolve(plot_df[plot_df['id']==id][f'collisions_{b}/shell'],
                                       np.ones(N) / N, mode='same')
          # plt.plot(scipy.signal.savgol_filter(df[f'collisions_ball{b}/shell'], 1001, 3))
          tt = saverate*np.arange(len(avg_collisions))
          plt.plot(tt, avg_collisions, color=colors[b])
          legend_str.append(f'{b}:' + labels[f'{b}'])
          plt.xlabel('Steps in env')
          all_avg_collisions[b].append(avg_collisions[:xlim])
        except: pass
    plt.legend(legend_str, bbox_to_anchor=[1.05, 1])
    plt.title(f'Collision rate ({w_k}k steps moving average): env {env_num}')
    plt.suptitle(f'Env{env_num}, {all_df["id"].unique()}')
    plt.tight_layout()
    plt.show()

    # Plot avg collisions across the cohort
    for b in ball_ids:
      if len(all_avg_collisions[b]) > 0:
        all_avg_collisions[b] = np.vstack(all_avg_collisions[b])
        mm = np.mean(all_avg_collisions[b], axis=0)
        ss = scipy.stats.sem(all_avg_collisions[b], axis=0)
        tt = saverate * np.arange(len(mm))
        plt.plot(tt, mm, color=colors[b])
        plt.fill_between(tt, mm-ss, mm+ss, alpha=0.5, color=colors[b])
        plt.xlabel('Steps in env')
        plt.title(f'Collision rate ({w_k}k step moving average): env {env_num}')
    plt.suptitle(f'Env{env_num}, {all_df["id"].unique()}')
    plt.show()
  print('done')


if __name__ == "__main__":
  main()