"""Summarize training metrics across multiple runs."""

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

def simple_axis(ax):
  # Hide the right and top spines
  ax.spines.right.set_visible(False)
  ax.spines.top.set_visible(False)

  # Only show ticks on the left and bottom spines
  ax.yaxis.set_ticks_position('left')
  ax.xaxis.set_ticks_position('bottom')

def load_metrics(id, remote, user_info, force_download = False):
  local_user = user_info['local_user']
  user = user_info['user']
  gcp_zone = user_info['gcp_zone']
  gcloud_path = user_info['gcloud_path']

  if remote[-1] == 'f':
    gcp_zone = 'us-central1-f'

  dir = f'/home/{local_user}/logs/{id}'
  fn = f'{dir}/metrics.jsonl'
  if force_download or not os.path.exists(fn):
    # First pull down the file
    os.makedirs(dir, exist_ok=True)
    print(f'Fetching log file from {remote}.')
    p = subprocess.Popen([gcloud_path, 'compute', 'scp',
                          f'{user}@{remote}:/home/{user}/logs/{id}/metrics.jsonl',
                          fn,
                          '--recurse', '--zone', gcp_zone,
                          '--project', 'hai-gcp-artificial'],
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    (output, err) = p.communicate()
    p_state = p.wait()
    print(output)

  df = pd.read_json(fn, lines=True)
  df['id'] = id
  return df

if __name__ == "__main__":
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

  savepath = '/home/saal2/logs/plots/'
  os.makedirs(savepath, exist_ok=True)

  force_download = False
  # ids = {
  #        'GEN-533': 't4-tf-2f',
  #        'GEN-534': 't4-tf-3f',
  #        'GEN-535': 't4-tf-4f',
  #        'GEN-543': 't4-tf-5',
  #        'GEN-537': 't4-tf-6f',
  #        'GEN-538': 't4-tf-7f',
  #        'GEN-539': 't4-tf-8f',
  #        'GEN-540': 't4-tf-9f',
  # }
  ids = {  # random policy playground
    'DRE-376': 't4-tf-2',
    'DRE-375': 't4-tf-1',
    'DRE-378': 't4-tf-2',
    'DRE-377': 't4-tf-1',
  }
  ids = {  # freerange, black
    'DRE-354': 't4-tf-2', # black
    'DRE-346': 't4-tf-8', # black
    'DRE-348': 't4-tf-9', # black
    # 'DRE-379': 't4-tf-4',  # black
    'DRE-475': 't4-tf-9', # when it is finished running....
  }
  ylim = [11264, 11272]
  xlim = [-1e5, 3e6]
  plt.figure()
  all_traces = []
  timepoints = []
  for id, remote in ids.items():
    df = load_metrics(id, remote, user_info, force_download=force_download)
    print(df)

    met = 'loss'
    print([x for x in df.keys() if met in x])

    # for met in ['ee/s-0/env-0/kl_loss', 'ee/env-0/image_loss']:
    for met in ['ee/env-0/image_loss']:
      timepoints = 2*df['ee/s-0/env-0/step'].dropna()
      plt.figure()
      plt.plot(timepoints, df[met].dropna())
      plt.suptitle(f"{df['id'].unique()[0]}:{met}")
      all_traces.append(df[met].dropna().to_numpy())
      plt.xlabel('steps')

  plt.ylim(ylim)
  plt.show()

  min_index = np.inf
  for i in range(len(all_traces)):
    if min_index > len(all_traces[i]):
      min_index = len(all_traces[i])

  plt.figure()
  for i in range(len(all_traces)):
    plt.plot(all_traces[i])
  plt.show()

  for i in range(len(all_traces)):
    all_traces[i] = all_traces[i][:min_index]

  stacked_traces = np.vstack(all_traces)
  m = np.mean(stacked_traces, axis=0)
  plt.plot(timepoints[:min_index], stacked_traces.T),
  plt.title(f'{met}')
  plt.suptitle(f'{list(ids.keys())}', fontsize=8)
  plt.xlabel('steps')
  plt.ylim(ylim)

  plt.show()

  tt = timepoints[:min_index]
  mm = np.mean(stacked_traces, axis=0)
  ss = scipy.stats.sem(stacked_traces, axis=0)
  plt.fill_between(tt, mm-ss, mm+ss, alpha=0.5)
  plt.plot(tt, mm)
  plt.title(f'Average {met}')
  plt.suptitle(f'{list(ids.keys())}', fontsize=8)
  plt.xlabel('steps')
  plt.ylim(ylim)
  plt.xlim(xlim)
  plt.ylabel('Image loss')
  simple_axis(plt.gca())
  plt.savefig(os.path.join(savepath, f'{list(ids.keys())}.pdf'))
  plt.show()

  d = np.diff(m)
  # ds = scipy.signal.savgol_filter(d, 51, 3)
  w = 100
  ds = np.convolve(d, np.ones(w)/w, mode='same')
  plt.plot(timepoints[:min_index-1], np.log(np.abs(ds)))
  plt.title(f'Smoothed change in {met} (w={w})')
  plt.suptitle(f'{list(ids.keys())}', fontsize=8)
  plt.xlabel('steps')
  plt.show()
  print('done')