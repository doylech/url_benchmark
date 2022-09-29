"""Summarize logs across multiple runs and multiple counterbalances, in a single plot. Final plot for baseline novelty runs."""

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


# def load_log(id, env_num, remote, user_info, force_download = False):
#   local_user = user_info['local_user']
#   user = user_info['user']
#   gcp_zone = user_info['gcp_zone']
#   gcloud_path = user_info['gcloud_path']
#   if remote[-1] == 'f':
#     gcp_zone = 'us-central1-f'
# 
#   fn = f'/home/{local_user}/logs/csv/log_{id}_train_env{env_num}.csv'
#   if force_download or not os.path.exists(fn):
#     # First pull down the file
#     print(f'Fetching log file from {remote}.')
#     p = subprocess.Popen([gcloud_path, 'compute', 'scp',
#                           f'{user}@{remote}:/home/{user}/logs/{id}/log_train_env{env_num}.csv',
#                           fn,
#                           '--recurse', '--zone', gcp_zone,
#                           '--project', 'hai-gcp-artificial'],
#                          stdin=subprocess.PIPE,
#                          stdout=subprocess.PIPE,
#                          stderr=subprocess.PIPE)
#     (output, err) = p.communicate()
#     p_state = p.wait()
#     print(output)
# 
#   df = pd.read_csv(fn)
#   df['id'] = id
#   df['env_num'] = env_num
#   return df

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

  plot_dir = f'{os.environ["HOME"]}/plots/novel_two_object'
  print(plot_dir)
  os.makedirs(plot_dir, exist_ok=True)


  all_ids = {}

  # id, remote, familiar_ball
  # all_ids['15M'] = { #15m
  #   'GEN-572': ['t4-tf-2', 'ball3'],
  #   'GEN-573': ['t4-tf-3', 'ball3'],
  #   'GEN-574': ['t4-tf-4', 'ball3'],
  #   'GEN-575': ['t4-tf-5f','ball3'],
  #   'GEN-576': ['t4-tf-6', 'ball4'],
  #   'GEN-577': ['t4-tf-7', 'ball4'],
  #   'GEN-579': ['t4-tf-8', 'ball4'],
  #   'GEN-580': ['t4-tf-9', 'ball4'],
  # }
  #
  # all_ids['10M'] = { #10m
  #   'GEN-588': ['t4-tf-2', 'ball3'], # 533
  #   'GEN-581': ['t4-tf-3', 'ball3'], # 534
  #   'GEN-582': ['t4-tf-4', 'ball3'], # 535
  #   'GEN-583': ['t4-tf-5f','ball3'], # 543
  #   'GEN-584': ['t4-tf-6', 'ball4'], # 537
  #   'GEN-585': ['t4-tf-7', 'ball4'], # 538
  #   'GEN-586': ['t4-tf-8', 'ball4'], # 539
  #   'GEN-587': ['t4-tf-9', 'ball4'], # 540
  # }
  #
  # all_ids['5M'] = { #5m
  #   'GEN-596': ['t4-tf-2', 'ball3'],
  #   'GEN-597': ['t4-tf-3', 'ball3'],
  #   'GEN-589': ['t4-tf-4', 'ball3'],
  #   'GEN-593': ['t4-tf-5f','ball3'],
  #   'GEN-591': ['t4-tf-6', 'ball4'],
  #   'GEN-592': ['t4-tf-7', 'ball4'],
  #   'GEN-594': ['t4-tf-8', 'ball4'],
  #   'GEN-595': ['t4-tf-9', 'ball4'],
  # }
  #
  # all_ids['1M'] = { #1m
  #   'GEN-603': ['t4-tf-2', 'ball3'], # 533
  #   'GEN-604': ['t4-tf-3', 'ball3'], # 534
  #   'GEN-605': ['t4-tf-4', 'ball3'], # 535
  #   'GEN-598': ['t4-tf-5f','ball3'], # 543
  #   'GEN-599': ['t4-tf-6', 'ball4'], # 537
  #   'GEN-600': ['t4-tf-7', 'ball4'], # 538
  #   'GEN-601': ['t4-tf-8', 'ball4'], # 539
  #   'GEN-602': ['t4-tf-9', 'ball4'], # 540
  # }

  # all_ids['20M'] = { #20m
  #   'GEN-606': ['t4-tf-2', 'ball3'],  # 533 yellow play
  #   'GEN-612': ['t4-tf-3', 'ball3'],  # 534 y
  #   'GEN-608': ['t4-tf-4', 'ball3'],  # 535 y
  #   'GEN-613': ['t4-tf-5f','ball3'],  # 543 y
  #   'GEN-609': ['t4-tf-6', 'ball4'],  # magenta play (537)
  #   'GEN-614': ['t4-tf-7', 'ball4'],  # 538 magenta
  #   'GEN-610': ['t4-tf-8', 'ball4'],  # 539 m
  #   'GEN-611': ['t4-tf-9', 'ball4'],  # 540 m
  # }

  # all_ids['SM'] = { #seek_magenta
  #   'GEN-948': ['t4-tf-2', 'ball3'],  # 533 yellow play
  #   'GEN-950': ['t4-tf-3', 'ball3'],  # 534 y
  #   # 'GEN-951': ['t4-tf-4', 'ball3'],  # 535 y
  #   # 'GEN-952': ['t4-tf-5', 'ball3'],  # 543 y
  #   'GEN-953': ['t4-tf-6', 'ball3'],  # magenta play (537)
  #   'GEN-954': ['t4-tf-7', 'ball3'],  # 538 magenta
  #   # 'GEN-955': ['t4-tf-8', 'ball3'],  # 539 m
  #   # 'GEN-956': ['t4-tf-9', 'ball3'],  # 540 m
  # }
  #
  # all_ids['SM_8x'] = { #seek_magenta 8x
  #   'GEN-959': ['t4-tf-2', 'ball3'],  # 533 yellow play
  #   'GEN-960': ['t4-tf-3', 'ball3'],  # 534 y
  #   'GEN-963': ['t4-tf-6', 'ball3'],  # magenta play (537)
  #   'GEN-962': ['t4-tf-7', 'ball3'],  # 538 magenta
  # }

  # all_ids['SY'] = { #seek_magenta
  #   'GEN-964': ['t4-tf-2', 'ball4'],  # 533 yellow play
  #   'GEN-965': ['t4-tf-3', 'ball4'],  # 534 y
  #   'GEN-966': ['t4-tf-4', 'ball4'],  # 535 y
  #   'GEN-967': ['t4-tf-5', 'ball4'],  # 543 y
  #   'GEN-968': ['t4-tf-6', 'ball4'],  # magenta play (537)
  #   'GEN-969': ['t4-tf-7', 'ball4'],  # 538 magenta
  #   'GEN-970': ['t4-tf-8', 'ball4'],  # 539 m
  #   'GEN-971': ['t4-tf-9', 'ball4'],  # 540 m
  # }
  #
  # all_ids['SC'] = { # seek_color
  #   'GEN-948': ['t4-tf-2', 'ball3'],  # 533 yellow play
  #   'GEN-950': ['t4-tf-3', 'ball3'],  # 534 y
  #   'GEN-951': ['t4-tf-4', 'ball3'],  # 535 y
  #   'GEN-952': ['t4-tf-5', 'ball3'],  # 543 y
  #   'GEN-968': ['t4-tf-6', 'ball4'],  # magenta play (537)
  #   'GEN-969': ['t4-tf-7', 'ball4'],  # 538 magenta
  #   'GEN-970': ['t4-tf-8', 'ball4'],  # 539 m
  #   'GEN-971': ['t4-tf-9', 'ball4'],  # 540 m
  # }
  #
  # all_ids['SC_8x'] = { # seek_color_8x
  #     'GEN-959': ['t4-tf-2', 'ball3'],  # 533 yellow play
  #     'GEN-960': ['t4-tf-3', 'ball3'],  # 534 y
  #     'GEN-984': ['t4-tf-4', 'ball3'],  # 535 y
  #     'GEN-985': ['t4-tf-5', 'ball3'],  # 543 y
  #     'GEN-986': ['t4-tf-6', 'ball4'],  # magenta play (537)
  #     'GEN-987': ['t4-tf-7', 'ball4'],  # 538 magenta
  #     'GEN-988': ['t4-tf-8', 'ball4'],  # 539 m
  #     'GEN-989': ['t4-tf-9', 'ball4'],  # 540 m
  # }

  # all_ids['p2e_8x_w&e'] = { # plan2explore 8x inner loop
  #   'GEN-972': ['t4-tf-2', 'ball3'],  # 533 yellow play
  #   'GEN-973': ['t4-tf-3', 'ball3'],  # 534 y
  #   'GEN-974': ['t4-tf-4', 'ball3'],  # 535 y
  #   'GEN-975': ['t4-tf-5', 'ball3'],  # 543 y
  #   'GEN-976': ['t4-tf-6', 'ball4'],  # magenta play (537)
  #   'GEN-977': ['t4-tf-7', 'ball4'],  # 538 magenta
  #   'GEN-978': ['t4-tf-8', 'ball4'],  # 539 m
  #   'GEN-979': ['t4-tf-9', 'ball4'],  # 540 m
  # }
  #
  # all_ids['rnd_recon_4x_w&e_resetac'] = {
  #   'DRE-43': ['t4-tf-4', 'ball3'], # gen 1001 y
  #   'DRE-39': ['t4-tf-5', 'ball3'], # gen 1008 y
  #   'DRE-40': ['t4-tf-8', 'ball4'], # gen 1006 m
  #   'DRE-41': ['t4-tf-9', 'ball4'], # gen 1007 m
  # }
  # #
  # all_ids['rnd_recon_8x_w&e_resetac'] = {
  #   'DRE-44': ['t4-tf-4', 'ball3'], # gen 1001 y
  #   'DRE-45': ['t4-tf-5', 'ball3'], # gen 1008 y
  #   'DRE-46': ['t4-tf-8', 'ball4'], # gen 1006 m
  #   'DRE-47': ['t4-tf-9', 'ball4'], # gen 1007 m
  # }
  #
  # all_ids['p2e_8x_w&e_resetac'] = {
  #   'DRE-48': ['t4-tf-2', 'ball3'], # gen 1001 y
  #   'DRE-49': ['t4-tf-3', 'ball3'], # gen 1008 y
  #   'DRE-50': ['t4-tf-6', 'ball4'], # gen 1006 m
  #   'DRE-51': ['t4-tf-7', 'ball4'], # gen 1007 m
  # }

  # all_ids['rnd_recon_4x_w&e_resetac'] = {
  #   'DRE-58': ['t4-tf-4', 'ball3'], # gen 1001 y
  #   'DRE-59': ['t4-tf-5', 'ball3'], # gen 1008 y
  #   'DRE-60': ['t4-tf-8', 'ball4'], # gen 1006 m
  #   'DRE-61': ['t4-tf-9', 'ball4'], # gen 1007 m
  # }

  # all_ids['rnd_recon_4x_w&e_resetac_15kprefill'] = {
  #   'DRE-62': ['t4-tf-4', 'ball3'], # gen 1001 y
  #   'DRE-63': ['t4-tf-5', 'ball3'], # gen 1008 y
  #   'DRE-64': ['t4-tf-8', 'ball4'], # gen 1006 m
  #   'DRE-65': ['t4-tf-9', 'ball4'], # gen 1007 m
  # }

  # all_ids['rnd_recon_4x_w&e_resetac_10kprefill_predtrainevery10'] = {
  #   'DRE-73': ['t4-tf-2', 'ball3'],  # 533 yellow play
  #   'DRE-74': ['t4-tf-3', 'ball3'],  # 534 y
  #   'DRE-75': ['t4-tf-4', 'ball3'],  # 535 y
  #   'DRE-76': ['t4-tf-5', 'ball3'],  # 543 y
  #   'DRE-77': ['t4-tf-6', 'ball4'],  # magenta play (537)
  #   'DRE-78': ['t4-tf-7', 'ball4'],  # 538 magenta
  #   'DRE-79': ['t4-tf-8', 'ball4'],  # 539 m
  #   'DRE-80': ['t4-tf-9', 'ball4'],  # 540 m
  # }

  # all_ids['p2e_16x_w&e_resetac_lr6e-4'] = {
  #   'DRE-89': ['t4-tf-2', 'ball3'],  # 533 yellow play
  #   'DRE-90': ['t4-tf-3', 'ball3'],  # 534 y
  #   'DRE-91': ['t4-tf-4', 'ball3'],  # 535 y
  #   'DRE-92': ['t4-tf-5', 'ball3'],  # 543 y
  #   'DRE-93': ['t4-tf-6', 'ball4'],  # magenta play (537)
  #   'DRE-94': ['t4-tf-7', 'ball4'],  # 538 magenta
  #   'DRE-95': ['t4-tf-8', 'ball4'],  # 539 m
  #   'DRE-96': ['t4-tf-9', 'ball4'],  # 540 m
  # }

  # all_ids['p2e_16x_w&e_resetac_lr3e-4'] = {
  #   'DRE-97': ['t4-tf-2', 'ball3'],  # 533 yellow play
  #   'DRE-98': ['t4-tf-3', 'ball3'],  # 534 y
  #   'DRE-99': ['t4-tf-4', 'ball3'],  # 535 y
  #   'DRE-100': ['t4-tf-5', 'ball3'],  # 543 y
  #   'DRE-101': ['t4-tf-6', 'ball4'],  # magenta play (537)
  #   'DRE-102': ['t4-tf-7', 'ball4'],  # 538 magenta
  #   'DRE-103': ['t4-tf-8', 'ball4'],  # 539 m
  #   'DRE-104': ['t4-tf-9', 'ball4'],  # 540 m
  # }

  # all_ids['1M_p2e_nondiscrete'] = { #1m
  #   'DRE-170': ['t4-tf-2', 'ball3'], # 533
  #   'DRE-171': ['t4-tf-3', 'ball3'], # 534
  #   'DRE-172': ['t4-tf-4', 'ball3'], # 535
  #   'DRE-173': ['t4-tf-5','ball3'], # 543
  #   'DRE-174': ['t4-tf-6', 'ball4'], # 537
  #   'DRE-175': ['t4-tf-7', 'ball4'], # 538
  #   'DRE-176': ['t4-tf-8', 'ball4'], # 539
  #   'DRE-177': ['t4-tf-9', 'ball4'], # 540
  # }

  # all_ids['1M_p2e_nondiscrete_2'] = { #1m
  #   'DRE-178': ['t4-tf-2', 'ball3'], # 533
  #   'DRE-179': ['t4-tf-3', 'ball3'], # 534
  #   'DRE-180': ['t4-tf-4', 'ball3'], # 535
  #   'DRE-181': ['t4-tf-5','ball3'], # 543
  #   'DRE-182': ['t4-tf-6', 'ball4'], # 537
  #   'DRE-183': ['t4-tf-7', 'ball4'], # 538
  #   'DRE-184': ['t4-tf-8', 'ball4'], # 539
  #   'DRE-185': ['t4-tf-9', 'ball4'], # 540
  # }

  # all_ids['1M_p2e_nondiscrete_3'] = { #1m
  #   'DRE-186': ['t4-tf-2', 'ball3'], #
  #   'DRE-187': ['t4-tf-3', 'ball3'], #
  #   'DRE-188': ['t4-tf-4', 'ball3'], #
  #   'DRE-189': ['t4-tf-5','ball3'], #
  #   'DRE-190': ['t4-tf-6', 'ball4'], #
  #   'DRE-191': ['t4-tf-7', 'ball4'], #
  #   'DRE-192': ['t4-tf-8', 'ball4'], #
  #   'DRE-193': ['t4-tf-8', 'ball4'], #
  # }

  # all_ids['1M_p2e_nondisc_inner8_reset_expl'] = { #1m
  #   'DRE-194': ['t4-tf-2', 'ball3'], #
  #   'DRE-195': ['t4-tf-3', 'ball3'], #
  #   'DRE-196': ['t4-tf-4', 'ball3'], #
  #   'DRE-197': ['t4-tf-5','ball3'], #
  #   'DRE-198': ['t4-tf-6', 'ball4'], #
  #   'DRE-199': ['t4-tf-7', 'ball4'], #
  #   'DRE-200': ['t4-tf-8', 'ball4'], #
  #   'DRE-201': ['t4-tf-9', 'ball4'], #
  # }

  # all_ids['1M_p2e_nondisc_inner8'] = { #1m
  #   'DRE-202': ['t4-tf-2', 'ball3'], #
  #   'DRE-203': ['t4-tf-3', 'ball3'], #
  #   'DRE-204': ['t4-tf-4', 'ball3'], #
  #   'DRE-205': ['t4-tf-5','ball3'], #
  #   'DRE-206': ['t4-tf-6', 'ball4'], #
  #   'DRE-207': ['t4-tf-7', 'ball4'], #
  #   'DRE-208': ['t4-tf-8', 'ball4'], #
  #   'DRE-209': ['t4-tf-9', 'ball4'], #
  # }

  # all_ids['1M_p2e_nondisc_singlebuff'] = { #1m
  #   'DRE-210': ['t4-tf-2', 'ball3'], #
  #   'DRE-216': ['t4-tf-3', 'ball3'], #
  #   'DRE-212': ['t4-tf-4', 'ball3'], #
  #   'DRE-214': ['t4-tf-5','ball3'], #
  #   'DRE-215': ['t4-tf-6', 'ball4'], #
  #   'DRE-217': ['t4-tf-7', 'ball4'], #
  #   'DRE-218': ['t4-tf-8', 'ball4'], #
  #   'DRE-219': ['t4-tf-9', 'ball4'], #
  # }

  # all_ids['1M_p2e_nondiscrete_4'] = { #1m
  #   'DRE-220': ['t4-tf-2', 'ball3'], #
  #   'DRE-221': ['t4-tf-3', 'ball3'], #
  #   'DRE-222': ['t4-tf-4', 'ball3'], #
  #   'DRE-223': ['t4-tf-5','ball3'], #
  #   'DRE-224': ['t4-tf-6', 'ball4'], #
  #   'DRE-225': ['t4-tf-7', 'ball4'], #
  #   'DRE-226': ['t4-tf-8', 'ball4'], #
  #   'DRE-227': ['t4-tf-9', 'ball4'], #
  # }

  # all_ids['1M_p2e_nondisc_innerwm_freeze_ensemble'] = { #1m
  #   'DRE-228': ['t4-tf-2', 'ball3'], #
  #   'DRE-229': ['t4-tf-3', 'ball3'], #
  #   'DRE-230': ['t4-tf-4', 'ball3'], #
  #   'DRE-231': ['t4-tf-5','ball3'], #
  #   'DRE-232': ['t4-tf-6', 'ball4'], #
  #   'DRE-233': ['t4-tf-7', 'ball4'], #
  #   'DRE-234': ['t4-tf-8', 'ball4'], #
  #   'DRE-235': ['t4-tf-9', 'ball4'], #
  # }

  # all_ids['1M_p2e_nondiscrete_5'] = { #1m
  #   'DRE-236': ['t4-tf-2', 'ball3'], #
  #   'DRE-237': ['t4-tf-3', 'ball3'], #
  #   'DRE-238': ['t4-tf-4', 'ball3'], #
  #   'DRE-239': ['t4-tf-5','ball3'], #
  #   'DRE-240': ['t4-tf-6', 'ball4'], #
  #   'DRE-241': ['t4-tf-7', 'ball4'], #
  #   'DRE-242': ['t4-tf-8', 'ball4'], #
  #   'DRE-243': ['t4-tf-9', 'ball4'], #
  # }

  # all_ids['1M_p2e_nondiscrete_6'] = { #1m
  #   'DRE-244': ['t4-tf-2', 'ball3'], #
  #   'DRE-245': ['t4-tf-3', 'ball3'], #
  #   'DRE-252': ['t4-tf-4', 'ball3'], #
  #   'DRE-247': ['t4-tf-5','ball3'], #
  #   'DRE-248': ['t4-tf-6', 'ball4'], #
  #   'DRE-253': ['t4-tf-7', 'ball4'], #
  #   'DRE-254': ['t4-tf-8', 'ball4'], #
  #   'DRE-251': ['t4-tf-9', 'ball4'], #
  # }

  all_ids['1M_p2e_nondiscrete_7'] = { #1m
    'DRE-257': ['t4-tf-2', 'ball3'], #
    'DRE-258': ['t4-tf-3', 'ball3'], #
    'DRE-259': ['t4-tf-4', 'ball3'], #
    'DRE-260': ['t4-tf-5','ball3'], #
    'DRE-261': ['t4-tf-6', 'ball4'], #
    'DRE-262': ['t4-tf-7', 'ball4'], #
    'DRE-263': ['t4-tf-8', 'ball4'], #
    'DRE-264': ['t4-tf-9', 'ball4'], #
  }
  colors = {
            3: 'orange',
            4: 'magenta',
            }

  colors_by_type = {
    'novel': 'red',
    'familiar': 'blue',
  }


  force_download = 0
  do_plot_indiv_expts = 0
  env_nums = [0]
  ball_ids = list(colors.keys())

  for expt_set, ids in all_ids.items():
    all_df = None
    changepoints = []
    for env_num in env_nums:
      labels = {}
      for id, vals in ids.items():
        remote = vals[0]
        familiar_ball = vals[1]
        df = au.load_log(id, env_num, remote, user_info, force_download=force_download)
        save_freq = np.diff(df['total_step'].to_numpy())[1]

        # Assign ball to novel or familiar
        labels[id] = {}
        for b in ball_ids:
          if f'ball{b}' == familiar_ball:
            df['familiar'] = df[f'collisions_ball{b}/shell']
            labels[id]['familiar'] = b
          else:
            df['novel'] = df[f'collisions_ball{b}/shell']
            labels[id]['novel'] = b

        if all_df is None:
          all_df = df
        else:
          all_df = pd.concat((all_df, df))

      # nf = np.array([0, 3e4, 6e4]) # Select time windows. 3e4 is 10 mins
      nf = np.array([0, 6e4]) # Select time windows. 3e4 is 10 mins
      if do_plot_indiv_expts:
        # Plot each individual expt
        for id in all_df['id'].unique():
          plot_df = all_df[all_df['id'] == id]

          plt.figure(figsize=(8, 3))
          plt.subplot(1,2,1)
          plt.plot(plot_df['agent0_xloc'], plot_df['agent0_yloc'], 'k.',
                   markersize=1, alpha=0.2)
          for b in ball_ids:
            try:
              plt.plot(plot_df[f'ball{b}_xloc'], plot_df[f'ball{b}_yloc'], '.',
                       markersize=2, color=colors[b], alpha=0.5)
            except: pass
          plt.title(f'Position {id}')

          plt.subplot(1,2,2)
          legend_str = []
          for b in ['familiar', 'novel']:
            try:
              plt.plot(np.cumsum(plot_df[b]),
                       color=colors[labels[id][b]])
              legend_str.append(b)
            except: pass
          # for b in ball_ids:
          #   try:
          #     # plt.plot(scipy.signal.savgol_filter(df[f'collisions_ball{b}/shell'], 1001, 3))
          #     plt.plot(np.cumsum(plot_df[f'collisions_ball{b}/shell']),
          #              color=colors[b])
          #     legend_str.append(f'ball{b}:' + labels[f'ball{b}'])
          #   except: pass
          [plt.axvline(x/20, linestyle='--', color='k') for x in nf[1:]]
          plt.legend(legend_str, bbox_to_anchor=[1.05, 1])
          plt.title(f'{expt_set}: Cumulative collisions: env {env_num}')
          plt.tight_layout()
          plt.show()

      # Now make a combined barchart
      for ii in range(len(nf)-1):
        nf1 = nf[ii]
        nf2 = nf[ii+1]
        env_df = all_df[all_df['env_num']==env_num]
        plot_df = env_df[(env_df['total_step'] > nf1) & (env_df['total_step'] < nf2)]
        plt.figure()
        all_means = []
        all_ball_means = []
        all_ball_x = []
        all_ball_color = []

        x_inds = []
        x_labels = []
        for b in ['familiar', 'novel']:
          x_ind = int(b=='novel')
          x_inds.append(x_ind)
          x_labels.append(b)
          ball_means = []
          ball_color = []
          for id in plot_df['id'].unique():
            vals = plot_df[plot_df['id']==id][b]
            ball_mean = np.mean(vals.to_numpy())
            ball_means.append(ball_mean)
            ball_color.append(colors[labels[id][b]])
          all_means.append(ball_means)
          all_ball_means.append(ball_means)
          all_ball_x.append([x_ind]*len(ball_means))
          all_ball_color.append(ball_color)
          plt.bar(x_ind, np.mean(ball_means), color=colors_by_type[b], alpha=0.2)

        # Now plot each expt
        for i in range(len(all_ball_means[0])):
          plt.plot(np.array(all_ball_x)[:, i], np.array(all_ball_means)[:, i], 'k', alpha=0.5)
          plt.plot(x_inds[0], all_ball_means[0][i], '.', color=all_ball_color[0][i], markersize=10)
          plt.plot(x_inds[1], all_ball_means[1][i], '.', color=all_ball_color[1][i], markersize=10)
        plt.xticks(x_inds, x_labels)
        # Get one-sided t-test (is familiar < novel)
        _, p = scipy.stats.ttest_rel(all_ball_means[0], all_ball_means[1], alternative='less')
        plt.annotate(f'familiar < novel: p={p:0.3f}', (0.7, 1.3))
        # plt.ylim([0, 1.8])
        plt.xlim([-0.5, 1.5])
        plt.title(f'{expt_set}: Mean collisions in {nf1:.0e} to {nf2:.0e} timesteps')
        plt.suptitle(f'Env{env_num}, {all_df["id"].unique()}', fontsize=8)
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.savefig(f'{plot_dir}/bar_{expt_set}.pdf')

        plt.show()

      # Plot all cumsum combined across expts
      # steps_to_plot = 1e5
      steps_to_plot = nf[1]*2
      skip = 20
      _CONTROL_TIMESTEP = .02 # s
      tt = np.arange(0, steps_to_plot, skip)*_CONTROL_TIMESTEP
      plt.figure()
      env_df = all_df[all_df['env_num'] == env_num]
      plot_df = env_df
      for id in plot_df['id'].unique():
        legend_str = []
        for b in ['familiar', 'novel']:
          nt = min(int(steps_to_plot/skip), len(plot_df[plot_df['id']==id]))
          try:
            plt.plot(tt[:nt],
                     np.cumsum(plot_df[plot_df['id']==id][b][:nt]),
                     color=colors_by_type[b])
            legend_str.append(b)
          except:
            print('Cumsum didnt work')
            pass
      plt.legend(legend_str, bbox_to_anchor=[1.05, 1])
      plt.title(f'{expt_set}: Cumulative collisions: env {env_num}')
      plt.suptitle(f'Env{env_num}, {all_df["id"].unique()}', fontsize=8)
      [plt.axvline(x*_CONTROL_TIMESTEP, linestyle='--', color='k') for x in nf[1:]]
      # plt.xlim([0, 1e5/20])
      plt.xlabel('Time (s)')
      plt.ylabel('Collisions with object')
      ax = plt.gca()
      ax.spines['right'].set_visible(False)
      ax.spines['top'].set_visible(False)
      ax.yaxis.set_ticks_position('left')
      ax.xaxis.set_ticks_position('bottom')
      plt.tight_layout()
      plt.savefig(f'{plot_dir}/cumsum_{expt_set}.pdf')
      plt.show()


  print('done')


if __name__ == "__main__":
  main()