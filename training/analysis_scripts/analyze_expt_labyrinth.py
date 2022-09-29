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
# from matplotlib.patches import Rectangle
import itertools

import training.analysis_scripts.analysis_utils as au
from training.analysis_scripts.labyrinth_utils import get_roi_visits, get_time_in_maze, get_turn_bias, organize_visits

CONTROL_TIMESTEP = 0.03 # This is based on the config value when the expt was run
def set_interactive_plot(interactive):
  import matplotlib as mpl
  mpl.use('TkAgg') if interactive else mpl.use('module://backend_interagg')


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

  # ids = {  # labyrinth
  #   'DRE-357': 't4-tf-6',
  #   'DRE-358': 't4-tf-7',
  #   'DRE-359': 't4-tf-8',
  #   'DRE-360': 't4-tf-9',
  # }
  ids = {  # labyrinth black
    'DRE-362': 't4-tf-10', #freerange
    # 'DRE-366': 't4-tf-9', #freerange # something wrong with this?
    # 'DRE-363': 't4-tf-6', #10k
    'DRE-364': 't4-tf-7', #10k
    # 'DRE-365': 't4-tf-8', #1k
    'DRE-380': 't4-tf-5', #100k
  }
  ids = {  # labyrinth black
    # 'DRE-381': 't4-tf-1', #10k
    # 'DRE-382': 't4-tf-2',  # 10k
    # 'DRE-390': 't4-tf-5',  # 100k
    # 'DRE-384': 't4-tf-6',  # 100k
    # 'DRE-385': 't4-tf-7',  # freerange
    # 'DRE-386': 't4-tf-8',  # freerange
    # 'DRE-387': 't4-tf-9',  # random
    # 'DRE-388': 't4-tf-10',  # random
    # 'DRE-389': 't4-tf-3',  # random
    'DRE-391': 't4-tf-1', # 100k
    'DRE-392': 't4-tf-2',  # 100k
    'DRE-394': 't4-tf-3',  # freerange
    'DRE-395': 't4-tf-5',  # freerange
  }

  ids = { # labyrinth black
    # 'DRE-442': 't4-tf-1',  # 100k
    # 'DRE-443': 't4-tf-2',  # 100k
    # 'DRE-444': 't4-tf-3',  # 100k
    # 'DRE-445': 't4-tf-4',  # 100k
    # 'DRE-446': 't4-tf-5',  # 200k
    # 'DRE-447': 't4-tf-6',  # 200k
    # 'DRE-448': 't4-tf-7',  # 200k
    # 'DRE-449': 't4-tf-8',  # 200k
    # # 'DRE-450': 't4-tf-9',  # 300k
    # 'DRE-451': 't4-tf-10',  # 300k
    'DRE-478': 't4-tf-2f',  # 50k
    # 'DRE-479': 't4-tf-3f',  # 50k
    # 'DRE-480': 't4-tf-4f',  # 50k
    # 'DRE-389': 't4-tf-3',  # random
    # 'DRA-20': 't4-tf-7',  # freerange
    # 'DRA-21': 't4-tf-8',  # freerange
    # 'DRA-22': 't4-tf-9',  # freerange
    # 'DRA-23': 't4-tf-10',  # freerange
    # 'DRA-24': 't4-tf-11',  # freerange
    # 'DRA-25': 't4-tf-12',  # freerange
  }


  labels = {
    'ball3': 'familiar',
    'ball4': 'novel',
  }
  colors = {
    'ball3': 'orange',
    'ball4': 'magenta',
  }

  ## TODO TODO
  do_plot_positions = True
  do_end_node_efficiency = True  # TODO: Plot total number of steps, and also steps in maze
  do_show_efficiency_from_start = False
  do_plot_node_sequence = True

  do_turn_bias = True
  do_time_in_maze = True
  do_simulate_random_turns = False

  force_download = 0

  env_nums = [0]

  saverate = 20
  # xlim = int(100e3 / saverate)
  # xlim = int(20000e3 / saverate)
  xlim = int(15000e3 / saverate)


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
      x = plot_df['agent0_xloc'][:xlim]
      y = plot_df['agent0_yloc'][:xlim]
      nt = len(x)
      (node_visits, intersection_visits,
       homecage_visits, stem_visits,
       right_visits, left_visits) = get_roi_visits(x, y, do_plot=True, do_plot_rect=True,
                                                   title_str=f'{id}: total steps: {saverate * len(x):.2E}')

      if do_time_in_maze:
        maze_occupancy = get_time_in_maze(homecage_visits[0], nt, 500, CONTROL_TIMESTEP, do_plot=True,
                                          title_str=f'{id} maze occupancy')

      if do_turn_bias:
        Psf, Pbf, Psa, Pbs = get_turn_bias(stem_visits, intersection_visits, right_visits, left_visits,
                                           nt, saverate, do_plot=True)

      ### Count num unique nodes, from the start
      if do_end_node_efficiency:
        # TODO: FACTOR THIS (and check that it is all okay) !!

        tt = saverate * np.arange(nt)
        node_vec = organize_visits(node_visits, nt)
        node_vec = node_vec[np.where(node_vec > 0)[0]]  # Get rid of all timepoints where not in any roi.

        node_seq = []
        last_node = 0
        for i in range(len(node_vec)):
          n = node_vec[i]
          if n != 0 and n != last_node:
            node_seq.append(n)
            last_node = n

        if do_plot_node_sequence:
          plt.plot(node_seq)
          plt.ylabel('Endnode id')
          plt.xlabel('Endnode visit')
          # simple_plot(plt.gca())
          plt.title(f'{id}: total steps: {saverate*nt:.2E}')


        # Count how many unique nodes have been visited, since the start.
        n_unique_0 = []
        for i in range(len(node_seq)):
          n_unique_0.append(len(np.unique(node_seq[:i])))

        # Count how many unique nodes have been visited in different sized windows.
        # In a string of n nodes, how many of these are distinct.
        # Slide a window of size n across the sequence, count d distinct nodes
        # in each window. Then average over d over all windows in all clips.
        mean_n_unique = {}
        ns = np.array([2, 4, 8, 16, 32, 48, 64, 100, 128, 156, 200, 250, 300, 350, 400, 450, 500, 512])
        for n in ns:
          n_unique = []
          for i in range(len(node_seq)-n):
            seq = node_seq[i:i+n]
            n_unique.append(len(np.unique(seq)))
          mean_n_unique[n] = np.mean(np.array(n_unique))

        N32 = np.inf
        for i in range(len(node_seq)):
          if len(np.unique(node_seq[:i])) == 32:
            N32 = i
        efficiency = 32/N32

        ### Numbers based on Fig 8 in https://elifesciences.org/articles/66175#equ1
        random_x = [2, 10, 20, 30, 60, 100, 200, 500, 1000, 2000]
        random_y = [2, 4.2, 6.4, 8.4, 13.7, 19.67, 31.6, 50, 62.6, 64]
        mousec1_x = [2, 10, 20, 30, 60, 100, 200, 500, 1000, 2000]
        mousec1_y = [2, 6.4, 11.9, 16.5, 28.1, 39.2, 53.3, 62.6, 63.5, 64]
        optimal = ns
        plt.figure()
        plt.semilogx(ns, optimal, 'k')
        plt.semilogx(random_x, random_y, 'b')
        plt.semilogx(mousec1_x, mousec1_y, 'r')
        plt.semilogx(ns, [mean_n_unique[x] for x in ns], 'm')
        if do_show_efficiency_from_start:
          plt.semilogx(np.arange(len(n_unique_0)), n_unique_0, 'g')
        plt.ylim([0, 64])
        plt.xlim([2, 2000])
        plt.axhline(32, color='k', linestyle='--')
        plt.ylabel('New end nodes found')
        plt.xlabel('End nodes visited')
        if do_show_efficiency_from_start:
          plt.legend(['Optimal (E=1.0)', 'Random (E=0.23)', 'Mouse C1 (E=0.42)', f'{id} (E={efficiency:0.2f})', f'{id} from start'])
        else:
          plt.legend(['Optimal (E=1.0)', 'Random (E=0.23)', 'Mouse C1 (E=0.42)', f'{id} (E={efficiency:0.2f})'])
        plt.title(f'Efficiency: {efficiency:0.2f}, total ends visited: {len(np.unique(node_seq))}, total steps: {saverate*nt:.2E}')
        plt.show()



    for id in all_df['id'].unique():
      plot_df = all_df[all_df['id'] == id]

      plt.figure(figsize=(8, 3))
      plt.subplot(1,2,1)
      plt.plot(plot_df['agent0_xloc'][:xlim], plot_df['agent0_yloc'][:xlim], 'k.',
               markersize=1, alpha=0.2)
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