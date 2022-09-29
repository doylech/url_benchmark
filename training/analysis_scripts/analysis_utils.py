"""Utils for analysis scripts."""

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

from common.plot import load_jsonl
from pathlib import Path

def load_log(id, env_num, remote, user_info, force_download = False):
  local_user = user_info['local_user']
  user = user_info['user']
  gcp_zone = user_info['gcp_zone']
  gcloud_path = user_info['gcloud_path']
  if remote[-1] == 'f':
    gcp_zone = 'us-central1-f'

  fn = f'/home/{local_user}/logs/csv/log_{id}_train_env{env_num}.csv'
  if force_download or not os.path.exists(fn):
    # First pull down the file
    print(f'Fetching log file from {remote}.')
    p = subprocess.Popen([gcloud_path, 'compute', 'scp',
                          f'{user}@{remote}:/home/{user}/logs/{id}/log_train_env{env_num}.csv',
                          fn,
                          '--recurse', '--zone', gcp_zone,
                          '--project', 'hai-gcp-artificial'],
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    (output, err) = p.communicate()
    p_state = p.wait()
    print(output)

  df = pd.read_csv(fn)
  df['id'] = id
  df['env_num'] = env_num
  return df

def load_metrics(id, env_num, remote, user_info, force_download = False):
  local_user = user_info['local_user']
  user = user_info['user']
  gcp_zone = user_info['gcp_zone']
  gcloud_path = user_info['gcloud_path']
  if remote[-1] == 'f':
    gcp_zone = 'us-central1-f'

  fn = f'/home/{local_user}/logs/metrics/metrics_{id}_train_env{env_num}.jsonl'
  if force_download or not os.path.exists(fn):
    # First pull down the file
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
  # df = load_jsonl(Path(fn))
  df['id'] = id
  df['env_num'] = env_num
  return df

