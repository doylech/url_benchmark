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

def simple_plot(ax):
  ax.spines.right.set_visible(False)
  ax.spines.top.set_visible(False)
  ax.yaxis.set_ticks_position('left')
  ax.xaxis.set_ticks_position('bottom')

import matplotlib
matplotlib.use('module://backend_interagg')

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

  all_ids = {}
  all_ids['Freerange'] = {
    'DRA-82': 't4-tf-1',  # freerange
    'DRA-83': 't4-tf-2',  # freerange
    'DRA-84': 't4-tf-3',  # freerange
    'DRA-85': 't4-tf-4',  # freerange
    'DRA-86': 't4-tf-5',  # freerange
    'DRA-87': 't4-tf-6',  # freerange
    'DRA-88': 't4-tf-7',  # freerange
    'DRA-89': 't4-tf-8',  # freerange
  }
  all_ids['Freerange_ckpt'] = {
    'DRA-90': 't4-tf-1',  # freerange
    'DRA-91': 't4-tf-2',  # freerange
    'DRA-92': 't4-tf-3',  # freerange
    'DRA-93': 't4-tf-4',  # freerange
    'DRA-94': 't4-tf-5',  # freerange
    'DRA-95': 't4-tf-6',  # freerange
    'DRA-96': 't4-tf-7',  # freerange
    'DRA-97': 't4-tf-8',  # freerange
  }
  all_ids['Freerange_ckpt_longer'] = {
    'DRA-98': 't4-tf-1',  # freerange
    'DRA-99': 't4-tf-2',  # freerange
    'DRA-100': 't4-tf-3',  # freerange
    'DRA-103': 't4-tf-6',  # freerange
    'DRA-104': 't4-tf-7',  # freerange
    'DRA-105': 't4-tf-8',  # freerange
  }

  all_ids['Freerange_ckpt16x'] = {
    'DRA-114': 't4-tf-1',  # freerange
    'DRA-115': 't4-tf-2',  # freerange
    'DRA-116': 't4-tf-3',  # freerange
    'DRA-117': 't4-tf-4',  # freerange
    'DRA-118': 't4-tf-5',  # freerange
    'DRA-119': 't4-tf-6',  # freerange
    'DRA-120': 't4-tf-7',  # freerange
    'DRA-121': 't4-tf-8',  # freerange
  }

  all_ids['seek_yellow'] = {
    'DRA-143': 't4-tf-1',  # freerange
    'DRA-144': 't4-tf-2',  # freerange
    'DRA-145': 't4-tf-3',  # freerange
    'DRA-146': 't4-tf-4',  # freerange
  }

  all_ids['seek_magenta'] = {
    'DRA-147': 't4-tf-5',  # freerange
    'DRA-148': 't4-tf-6',  # freerange
    'DRA-149': 't4-tf-7',  # freerange
    'DRA-150': 't4-tf-8',  # freerange
  }
  all_ids['seek_magenta24x'] = {
    'DRA-159': 't4-tf-1',  # freerange
    'DRA-160': 't4-tf-2',  # freerange
    'DRA-161': 't4-tf-3',  # freerange
    'DRA-162': 't4-tf-4',  # freerange
  }
  all_ids['seek_yellow24x'] = {
    'DRA-163': 't4-tf-5',  # freerange
    'DRA-164': 't4-tf-6',  # freerange
    'DRA-165': 't4-tf-7',  # freerange
    'DRA-166': 't4-tf-8',  # freerange
  }
  all_ids['reset_obj'] = {
    'DRA-151': 't4-tf-1',  # freerange
    'DRA-152': 't4-tf-2',  # freerange
    'DRA-153': 't4-tf-3',  # freerange
    'DRA-154': 't4-tf-4',  # freerange
    'DRA-155': 't4-tf-5',  # freerange
    'DRA-156': 't4-tf-6',  # freerange
    'DRA-157': 't4-tf-7',  # freerange
    'DRA-158': 't4-tf-8',  # freerange
  }

  labels = {
    'object1': 'familiar',
    'object2': 'novel',
    # 'object3': 'object3',
    # 'object4': 'object4',
  }
  colors = {
    'object1': 'orange',
    'object2': 'magenta',
    # 'object3': 'orange',
    # 'object4': 'magenta',
  }

  # which_expts = ['Freerange', 'Position reset (1k)',
  #                'Freerange + clear buffer', 'Position reset (1k) + clear buffer']
  # which_expts = ['Prioritized temporal']
  which_expts = ['seek_magenta24x']
  # which_expts = ['Position reset (1k)']


  force_download = 0
  env_nums = [0]
  saverate = 20

  which_chunk = 'novel'

  prefill = 5e3 # prefill

  if which_chunk == 'play':
    tstart = int(0/saverate)
    tend = int(3e4/saverate)
  if which_chunk == 'play_long':
    tstart = int(0/saverate)
    tend = int(6e4/saverate)
  elif which_chunk == 'novel':
    tstart = int(3e4/saverate)
    tend = int(6e4/saverate)
  elif which_chunk == 'novel_long':
    tstart = int(6e4/saverate)
    tend = int(9e4/saverate)
  elif which_chunk == 'all':
    tstart = int(0/saverate)
    tend = int(6e4/saverate)

  tstart += int(prefill/saverate)
  tend += int(prefill/saverate)

  # nf = save_freq*1e3*np.array([0, 5, 10, 15, 20, 25])
  # nf = np.array([0, 3e4, 6e4])
  # nf = np.array([0, 6e4])
  nf = np.array([0, 3e4]) # Which chunks to quantify

  do_plot_each_expt = True
  do_summary_barplot = True
  do_cumsum_all_expts = True
  do_cumsum_across_expts = True

  do_avg_collisions_all_expts = False
  do_avg_collisions_across_expts = False

  ball_ids = list(colors.keys())

  # for expt_set, ids in all_ids.items():
  for expt_set in which_expts:
    ids = all_ids[expt_set]

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

      if do_plot_each_expt:
        for id in all_df['id'].unique():
          plot_df = all_df[all_df['id'] == id]

          plt.figure(figsize=(8, 3))
          ax = plt.subplot(1,2,1)
          plt.plot(plot_df['agent0_xloc'][tstart:tend], plot_df['agent0_yloc'][tstart:tend], 'k.',
                   markersize=1, alpha=0.2)
          for b in ball_ids:
            try:
              plt.plot(plot_df[f'{b}_xloc'][tstart:tend], plot_df[f'{b}_yloc'][tstart:tend], '.',
                       markersize=2, color=colors[b], alpha=0.5)
            except: pass
          plt.title(f'Position {id}')
          ax.axis('equal')
          plt.xlim(-15, 15)
          plt.ylim(-15, 15)

          plt.subplot(1,2,2)
          legend_str = []
          for b in ball_ids:
            try:
              # plt.plot(scipy.signal.savgol_filter(df[f'collisions_ball{b}/shell'], 1001, 3))
              tt = saverate*np.arange(np.cumsum(plot_df[f'collisions_{b}/shell'][tstart:tend]).shape[0])
              plt.plot(tt, np.cumsum(plot_df[f'collisions_{b}/shell'][tstart:tend]),
                       color=colors[b])
              legend_str.append(f'{b}:' + labels[f'{b}'])
              plt.xlabel('Steps in env')
              plt.ylabel('Cumulative collisions')
            except: pass
          # [plt.axvline(x/saverate, linestyle='--', color='k') for x in nf[1:]]
          [plt.axvline(x, linestyle='--', color='k') for x in nf[1:]]
          plt.legend(legend_str, bbox_to_anchor=[1.05, 1])
          plt.suptitle(f'env {env_num}', fontsize=6)
          plt.title(f'{expt_set}: Cumulative collisions')
          plt.tight_layout()
          plt.show()


      if do_summary_barplot:
        for ii in range(len(nf)-1):
          nf1 = nf[ii] + tstart*saverate
          nf2 = nf[ii+1] + tstart*saverate
          env_df = all_df[all_df['env_num']==env_num]
          plot_df = env_df[(env_df['total_step'] > nf1) & (env_df['total_step'] < nf2)]
          plt.figure()
          all_means = []
          all_ball_means = []
          all_ball_x = []
          for ind, b in enumerate(ball_ids):
            try:
              ball_means = []
              for id in plot_df['id'].unique():
                  vals = plot_df[plot_df['id']==id][f'collisions_{b}/shell']
                  ball_means.append(np.mean(vals.to_numpy()))

              all_means.append(ball_means)
              all_ball_means.append(ball_means)
              all_ball_x.append([ind]*len(ball_means))
              plt.plot([ind]*len(ball_means), ball_means, '.', color=colors[b])
              plt.bar(ind, np.mean(ball_means), color=colors[b], alpha=0.2)
            except:
              pass

          _, p = scipy.stats.wilcoxon(all_ball_means[0], all_ball_means[1])

          plt.plot(np.array(all_ball_x), np.array(all_ball_means), 'k')
          plt.xticks(np.arange(len(ball_ids)), [labels[f'{b}'] for b in ball_ids])
          plt.ylabel('Mean collisions')
          plt.title(f'{expt_set}: timesteps [{nf1:.2e} to {nf2:.2e}], wilcoxon p={p:0.3f}')
          plt.suptitle(f'Env{env_num}, {all_df["id"].unique()}', fontsize=6)
          simple_plot(plt.gca())
          plt.tight_layout()
          plt.show()

      # Plot all cumsum combined across expts
      if do_cumsum_all_expts:
        all_cumsum_collisions = {}
        plt.figure()
        env_df = all_df[all_df['env_num'] == env_num]
        plot_df = env_df
        for b in ball_ids:
          all_cumsum_collisions[b] = []
        for id in plot_df['id'].unique():
          legend_str = []
          legend_plots = []
          for b in ball_ids:
            try:
              # plt.plot(scipy.signal.savgol_filter(df[f'collisions_ball{b}/shell'], 1001, 3))
              tt = saverate*np.arange(len(plot_df[plot_df['id']==id][f'collisions_{b}/shell']))[tstart:tend]
              cumsum_collisions = np.cumsum(plot_df[plot_df['id']==id][f'collisions_{b}/shell'][tstart:tend])
              print(len(cumsum_collisions))
              p1, = plt.plot(tt, cumsum_collisions,
                       color=colors[b])
              legend_str.append(f'{b}:' + labels[f'{b}'])
              legend_plots.append(p1)
              plt.xlabel('Steps in env')
              plt.ylabel('Cumulative collisions')
              all_cumsum_collisions[b].append(cumsum_collisions)
            except: pass
        plt.legend(legend_plots, legend_str, bbox_to_anchor=[1.05, 1], frameon=False)
        plt.title(f'{expt_set}: Cumulative collisions (n={len(plot_df["id"].unique())})')
        plt.suptitle(f'Env{env_num}, {all_df["id"].unique()}', fontsize=6)
        simple_plot(plt.gca())
        plt.tight_layout()
        plt.show()

        if do_cumsum_across_expts:
          # min_t = min([len(x) for x in all_cumsum_collisions])
          # all_cumsum_collisions = [x[:min_t] for x in all_cumsum_collisions]

          plt.figure()
          legend_str = []
          legend_plots = []
          # Plot avg collisions across the cohort
          for b in ball_ids:
            if len(all_cumsum_collisions[b]) > 0:
              all_cumsum_collisions[b] = np.vstack(all_cumsum_collisions[b])
              mm = np.mean(all_cumsum_collisions[b], axis=0)
              ss = scipy.stats.sem(all_cumsum_collisions[b], axis=0)
              tt = saverate * np.arange(len(mm))
              p1, = plt.plot(tt, mm, color=colors[b])
              plt.fill_between(tt, mm - ss, mm + ss, alpha=0.5, color=colors[b])
              plt.xlabel('Steps in env')
              plt.ylabel('Cumulative collisions')
              legend_str.append(f'{b}:' + labels[f'{b}'])
              legend_plots.append(p1)
              plt.title(f'{expt_set}: Cumulative collisions (n={len(plot_df["id"].unique())})')
          # plt.legend(legend_plots, legend_str, bbox_to_anchor=[1.05, 1], frameon=False)
          plt.legend(legend_plots, legend_str, frameon=False, loc='upper left')
          plt.suptitle(f'Env{env_num}, {all_df["id"].unique()}', fontsize=6)
          simple_plot(plt.gca())
          plt.tight_layout()
          plt.show()
        print('done')

      if do_avg_collisions_all_expts:
        plt.figure()
        env_df = all_df[all_df['env_num'] == env_num]
        plot_df = env_df
        all_avg_collisions = {}
        for b in ball_ids:
          all_avg_collisions[b] = []
        for id in plot_df['id'].unique():
          legend_str = []
          legend_plots = []
          for b in ball_ids:
            try:
              w_k = 5
              N = int(w_k*1e3/saverate)
              avg_collisions = np.convolve(plot_df[plot_df['id']==id][f'collisions_{b}/shell'][tstart:tend],
                                           np.ones(N) / N, mode='same')
              # avg_collisions = plot_df[plot_df['id']==id][f'collisions_{b}/shell'][tstart:tend]
              # plt.plot(scipy.signal.savgol_filter(df[f'collisions_ball{b}/shell'], 1001, 3))
              tt = saverate*np.arange(len(avg_collisions))
              p1, = plt.plot(tt, avg_collisions, color=colors[b])
              legend_str.append(f'{b}:' + labels[f'{b}'])
              legend_plots.append(p1)
              plt.xlabel('Steps in env')
              plt.ylabel('Collision rate')
              all_avg_collisions[b].append(avg_collisions)
            except: pass
        plt.legend(legend_plots, legend_str, bbox_to_anchor=[1.05, 1], frameon=False)
        plt.title(f'{expt_set}: Collision rate ({w_k}k step moving average) (n={len(plot_df["id"].unique())})')
        plt.suptitle(f'Env{env_num}, {all_df["id"].unique()}', fontsize=6)
        simple_plot(plt.gca())
        plt.tight_layout()
        plt.show()

        if do_avg_collisions_across_expts:
          # Plot avg collisions across the cohort
          legend_str = []
          legend_plots = []
          plt.figure()
          for b in ball_ids:
            if len(all_avg_collisions[b]) > 0:
              all_avg_collisions[b] = np.vstack(all_avg_collisions[b])
              mm = np.mean(all_avg_collisions[b], axis=0)
              ss = scipy.stats.sem(all_avg_collisions[b], axis=0)
              tt = saverate * np.arange(len(mm))
              p1, = plt.plot(tt, mm, color=colors[b])
              plt.fill_between(tt, mm-ss, mm+ss, alpha=0.5, color=colors[b])
              legend_str.append(f'{b}:' + labels[f'{b}'])
              legend_plots.append(p1)
              plt.xlabel('Steps in env')
              plt.ylabel('Collision rate')
              plt.title(f'{expt_set}: Collision rate ({w_k}k step moving average) (n={len(plot_df["id"].unique())})')
          plt.legend(legend_plots, legend_str, bbox_to_anchor=[1.05, 1], frameon=False)
          plt.suptitle(f'Env{env_num}, {all_df["id"].unique()}', fontsize=6)
          simple_plot(plt.gca())
          plt.tight_layout()
          plt.show()
        print('done')


if __name__ == "__main__":
  main()