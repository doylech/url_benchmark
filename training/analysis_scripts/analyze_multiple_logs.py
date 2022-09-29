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


#   # ids = {'GEN-367': 't4-tf-6f',
#   #        'GEN-368': 't4-tf-7f',
#   #        'GEN-369': 't4-tf-8f',
#   #        'GEN-370': 't4-tf-9f',
#   #        }
#   # labels = {'ball2': 'novel',
#   #           'ball3': 'familiar',
#   #           'ball4': 'familiar',
#   #           'ball5': 'familiar'}
#   #
#   # ids = {'GEN-363': 't4-tf-6f',
#   #        'GEN-364': 't4-tf-7f',
#   #        'GEN-365': 't4-tf-8f',
#   #        'GEN-366': 't4-tf-9f',
#   #        }
#   # labels = {'ball2': 'familiar',
#   #           'ball3': 'familiar',
#   #           'ball4': 'novel',
#   #           'ball5': 'familiar'}
#
#   # ids = {'GEN-379': 't4-tf-6f',
#   #        'GEN-380': 't4-tf-7f',
#   #        'GEN-381': 't4-tf-8f',
#   #        'GEN-382': 't4-tf-9f',
#   #        }
#   # ids = {'GEN-391': 't4-tf-6f',
#   #        'GEN-392': 't4-tf-7f',
#   #        'GEN-393': 't4-tf-8f',
#   #        'GEN-394': 't4-tf-9f',
#   #        }
#   # labels = {'ball2': 'familiar',
#   #           'ball3': 'familiar',
#   #           'ball4': 'familiar',
#   #           'ball5': 'novel'}
#   #
#   # ids = {'GEN-395': 't4-tf-6f',
#   #        'GEN-396': 't4-tf-7f',
#   #        'GEN-397': 't4-tf-8f',
#   #        'GEN-398': 't4-tf-9f',
#   #        }
#   # labels = {'ball2': 'familiar',
#   #           'ball3': 'familiar',
#   #           'ball4': 'familiar',
#   #           'ball5': 'familiar'}
#   #
#   # ids = {'GEN-401': 't4-tf-6f',
#   #        'GEN-402': 't4-tf-7f',
#   #        'GEN-403': 't4-tf-8f',
#   #        'GEN-404': 't4-tf-9f',
#   #        }
#   # labels = {'ball2': 'familiar',
#   #           'ball3': 'familiar',
#   #           'ball4': 'familiar',
#   #           'ball5': 'novel'}
#   #
#   # colors = {2: 'red',
#   #           3: 'orange',
#   #           4: 'purple',
#   #           5: 'blue'}
#   #
#   # ids = {'GEN-414': 't4-tf-6f',
#   #        'GEN-415': 't4-tf-7f',
#   #        'GEN-416': 't4-tf-8f',
#   #        'GEN-417': 't4-tf-9f',
#   #        }
#   # labels = {
#   #           'ball3': 'familiar',
#   #           'ball4': 'novel',
#   #           }
#   #
#   # ids = {'GEN-418': 't4-tf-2f',
#   #        'GEN-419': 't4-tf-3f',
#   #        'GEN-420': 't4-tf-4f',
#   #        'GEN-421': 't4-tf-5f',
#   #        }
#   # labels = {
#   #           'ball3': 'novel',
#   #           'ball4': 'familiar',
#   #           }
#
#   # ids = {'GEN-433': 't4-tf-2f',
#   #        'GEN-434': 't4-tf-3f',
#   #        'GEN-435': 't4-tf-4f',
#   #        'GEN-421': 't4-tf-5f',
#   #        }
#   # labels = {
#   #           'ball3': 'novel',
#   #           'ball4': 'familiar',
#   #           }
#
#   # TODO BELOW ###
#   # ids = {'GEN-433': 't4-tf-6',
#   #        'GEN-434': 't4-tf-7',
#   #        'GEN-435': 't4-tf-8',
#   #        }
#   # labels = {
#   #           'ball3': 'novel',
#   #           'ball4': 'familiar',
#   #           }
#
#   # ids = {'GEN-430': 't4-tf-3',
#   #        'GEN-431': 't4-tf-4',
#   #        'GEN-432': 't4-tf-5',
#   #        }
#   # labels = {
#   #           'ball3': 'familiar',
#   #           'ball4': 'novel',
#   #           }
#   #
#   # ids = {'GEN-426': 't4-tf-6f',
#   #        'GEN-427': 't4-tf-7f',
#   #        'GEN-428': 't4-tf-8f',
#   #        'GEN-429': 't4-tf-9f',
#   #        }
#   # labels = {
#   #           'ball3': 'familiar',
#   #           'ball4': 'novel',
#   #           }
#   #
#   # ids = {'GEN-422': 't4-tf-2f',
#   #        'GEN-423': 't4-tf-3f',
#   #        'GEN-424': 't4-tf-4f',
#   #        'GEN-425': 't4-tf-5f',
#   #        }
#   # labels = {
#   #           'ball3': 'novel',
#   #           'ball4': 'familiar',
#   #           }
#
#   # ids = {#'GEN-436': 't4-tf-2f',
#   #        'GEN-438': 't4-tf-3f',
#   #        'GEN-439': 't4-tf-4f',
#   #        'GEN-440': 't4-tf-5f',
#   #        }
#   # labels = {
#   #           'ball3': 'novel',
#   #           'ball4': 'familiar',
#   #           }
#   #
#   # ids = {'GEN-441': 't4-tf-6f',
#   #        'GEN-442': 't4-tf-7f',
#   #        'GEN-443': 't4-tf-8f',
#   #        'GEN-444': 't4-tf-9f',
#   #        }
#   # labels = {
#   #           'ball3': 'familiar',
#   #           'ball4': 'novel',
#   #           }
#
#   # ids = {
#   #        'GEN-462': 't4-tf-8',
#   #        'GEN-463': 't4-tf-7',
#   #       'GEN-464': 't4-tf-6',
#   #       'GEN-465': 't4-tf-5',
#   #        }
#   # labels = {
#   #   'ball3': 'novel',
#   #   'ball4': 'familiar',
#   # }
#
#   # ids = {
#   #        'GEN-468': 't4-tf-3',
#   #        'GEN-467': 't4-tf-4',
#   #       'GEN-466': 't4-tf-5',
#   #        }
#   # labels = {
#   #   'ball3': 'familiar',
#   #   'ball4': 'novel',
#   # }
#
#
#   # ids = {
#   #        'GEN-471': 't4-tf-3',
#   #        'GEN-470': 't4-tf-4',
#   #       'GEN-469': 't4-tf-5',
#   #        }
#   # labels = {
#   #   'ball3': 'familiar',
#   #   'ball4': 'novel',
#   # }
#
#   # ids = {
#   #        'GEN-471': 't4-tf-3',
#   #        'GEN-470': 't4-tf-4',
#   #       'GEN-469': 't4-tf-5',
#   #        }
#   # labels = {
#   #   'ball3': 'familiar',
#   #   'ball4': 'novel',
#   # }
#
# #############
#   # ids = {
#   #        'GEN-486': 't4-tf-9',
#   #        'GEN-485': 't4-tf-8',
#   #        'GEN-484': 't4-tf-7',
#   #        'GEN-483': 't4-tf-6',
#   # }
#   # labels = {
#   #   'ball3': 'novel',
#   #   'ball4': 'familiar',
#   # }
#
#   # ids = {
#   #        'GEN-482': 't4-tf-5',
#   #        'GEN-481': 't4-tf-4',
#   #        'GEN-480': 't4-tf-3',
#   #        'GEN-479': 't4-tf-2',
#   # }
#   # labels = {
#   #   'ball3': 'familiar',
#   #   'ball4': 'novel',
#   # }
#
#   # ids = {
#   #        'GEN-457': 't4-tf-9f',
#   #        'GEN-456': 't4-tf-8f',
#   #        'GEN-455': 't4-tf-7f',
#   # }
#   # labels = {
#   #   'ball3': 'familiar',
#   #   'ball4': 'novel',
#   # }
#
#   # ids = {
#   #        'GEN-454': 't4-tf-6f',
#   #        'GEN-453': 't4-tf-5f',
#   #        'GEN-452': 't4-tf-4f',
#   # }
#   # labels = {
#   #   'ball3': 'novel',
#   #   'ball4': 'familiar',
#   # }
#
#   # ids = {
#   #        'GEN-494': 't4-tf-9',
#   #        'GEN-493': 't4-tf-8',
#   #        'GEN-492': 't4-tf-7',
#   #        'GEN-491': 't4-tf-6',
#   # }
#   # labels = {
#   #   'ball3': 'novel',
#   #   'ball4': 'familiar',
#   # }
#   #
#   # ids = {
#   #        'GEN-490': 't4-tf-5',
#   #        'GEN-489': 't4-tf-4',
#   #        'GEN-488': 't4-tf-3',
#   #        'GEN-487': 't4-tf-2',
#   # }
#   # labels = {
#   #   'ball3': 'familiar',
#   #   'ball4': 'novel',
#   # }
#
#   # ids = {
#   #        'GEN-495': 't4-tf-4f',
#   #        'GEN-496': 't4-tf-4f',
#   #        'GEN-497': 't4-tf-4f',
#   #        'GEN-498': 't4-tf-4f',
#   #        'GEN-499': 't4-tf-4f',
#   # }
#   # labels = {
#   #   'ball3': 'novel',
#   #   'ball4': 'familiar',
#   # }
#
#   # ids = {
#   #        'GEN-500': 't4-tf-4f',
#   #        'GEN-501': 't4-tf-5f',
#   #        'GEN-502': 't4-tf-6f',
#   # }
#   # labels = {
#   #   'ball3': 'novel',
#   #   'ball4': 'familiar',
#   # }
#   # ids = {
#   #        'GEN-503': 't4-tf-7f',
#   #        'GEN-504': 't4-tf-8f',
#   #        'GEN-505': 't4-tf-9f',
#   # }
#   # labels = {
#   #   'ball3': 'familiar',
#   #   'ball4': 'novel',
#   # }
#
#   # ids = {
#   #        'GEN-507': 't4-tf-4f',
#   #        'GEN-508': 't4-tf-5f',
#   #        'GEN-513': 't4-tf-6f',
#   # }
#   # labels = {
#   #   'ball3': 'novel',
#   #   'ball4': 'familiar',
#   # }
#   # ids = {
#   #        'GEN-510': 't4-tf-7f',
#   #        'GEN-511': 't4-tf-8f',
#   #        'GEN-512': 't4-tf-9f',
#   # }
#   # labels = {
#   #   'ball3': 'familiar',
#   #   'ball4': 'novel',
#   # }
#
#   # ids = {
#   #        'GEN-515': 't4-tf-4f',
#   #        'GEN-516': 't4-tf-5f',
#   #        'GEN-517': 't4-tf-6f',
#   # }
#   # labels = {
#   #   'ball3': 'novel',
#   #   'ball4': 'familiar',
#   # }
#   # ids = {
#   #        'GEN-518': 't4-tf-7f',
#   #        'GEN-519': 't4-tf-8f',
#   #        'GEN-520': 't4-tf-9f',
#   # }
#   # labels = {
#   #   'ball3': 'familiar',
#   #   'ball4': 'novel',
#   # }
#
#   # ids = {
#   #        'GEN-521': 't4-tf-4f',
#   #        'GEN-522': 't4-tf-5f',
#   #        'GEN-523': 't4-tf-6f',
#   # }
#   # labels = {
#   #   'ball3': 'novel',
#   #   'ball4': 'familiar',
#   # }
#   # ids = {
#   #        'GEN-524': 't4-tf-7f',
#   #        'GEN-525': 't4-tf-8f',
#   #        'GEN-526': 't4-tf-9f',
#   # }
#   # labels = {
#   #   'ball3': 'familiar',
#   #   'ball4': 'novel',
#   # }
#
#   # ids = {
#   #        'GEN-527': 't4-tf-4f',
#   #        'GEN-528': 't4-tf-5f',
#   #        'GEN-529': 't4-tf-6f',
#   # }
#   # labels = {
#   #   'ball3': 'novel',
#   #   'ball4': 'familiar',
#   # }
#   # ids = {
#   #        'GEN-530': 't4-tf-7f',
#   #        'GEN-531': 't4-tf-8f',
#   #        'GEN-532': 't4-tf-9f',
#   # }
#   # labels = {
#   #   'ball3': 'familiar',
#   #   'ball4': 'novel',
#   # }
#   #
#   # ids = {
#   #        'GEN-545': 't4-tf-8',
#   # }
#   # labels = {
#   #   'ball3': 'familiar',
#   #   'ball4': 'novel',
#   # }
#   # ids = {
#   #        # 'GEN-548': 't4-tf-2',
#   #       'GEN-549': 't4-tf-2',
#   # }
#   # labels = {
#   #   'ball3': 'novel',
#   #   'ball4': 'familiar',
#   # }
#   #
#   # ###################
#   #
#   # ids = { #1m
#   #       'GEN-550': 't4-tf-2',
#   #       'GEN-551': 't4-tf-3',
#   #       'GEN-552': 't4-tf-4',
#   # }
#   # labels = {
#   #   'ball3': 'familiar',
#   #   'ball4': 'novel',
#   # }
#   #
#   # # ids = { # 1m
#   # #   'GEN-553': 't4-tf-6',
#   # #   'GEN-554': 't4-tf-7',
#   # #   'GEN-557': 't4-tf-9',
#   # #   'GEN-558': 't4-tf-8',
#   # #
#   # # }
#   # # labels = {
#   # #   'ball3': 'novel',
#   # #   'ball4': 'familiar',
#   # # }
#   #
#   # # ids = { #8m
#   # #   'GEN-565': 't4-tf-2',
#   # #   'GEN-566': 't4-tf-3',
#   # #   'GEN-559': 't4-tf-4',
#   # # }
#   # # labels = {
#   # #   'ball3': 'familiar',
#   # #   'ball4': 'novel',
#   # # }
#   # #
#   # # ids = { # 8m
#   # #   'GEN-561': 't4-tf-6',
#   # #   'GEN-562': 't4-tf-7',
#   # #   'GEN-563': 't4-tf-8',
#   # #   'GEN-564': 't4-tf-9',
#   # #
#   # # }
#   # # labels = {
#   # #   'ball3': 'novel',
#   # #   'ball4': 'familiar',
#   # # }
#   # #
#   #
#   # # ids = { #7m
#   # #   'GEN-567': 't4-tf-2',
#   # #   'GEN-568': 't4-tf-3',
#   # # }
#   # # labels = {
#   # #   'ball3': 'familiar',
#   # #   'ball4': 'novel',
#   # # }
#   #
#   # # ids = { # 7m
#   # #   'GEN-569': 't4-tf-7',
#   # #   'GEN-570': 't4-tf-8',
#   # #   'GEN-571': 't4-tf-9',
#   # #
#   # # }
#   # # labels = {
#   # #   'ball3': 'novel',
#   # #   'ball4': 'familiar',
#   # # }
#   #
#   # # ids = { #15m
#   # #   'GEN-572': 't4-tf-2',
#   # #   'GEN-573': 't4-tf-3',
#   # #   'GEN-574': 't4-tf-4',
#   # #   'GEN-575': 't4-tf-5f',
#   # # }
#   # # labels = {
#   # #   'ball3': 'familiar',
#   # #   'ball4': 'novel',
#   # # }
#   #
#   # # ids = { #15m
#   # #   'GEN-576': 't4-tf-6',
#   # #   'GEN-577': 't4-tf-7',
#   # #   'GEN-579': 't4-tf-8',
#   # #   'GEN-580': 't4-tf-9',
#   # # }
#   # # labels = {
#   # #   'ball3': 'novel',
#   # #   'ball4': 'familiar',
#   # # }
#   #
#   # ids = { #10m
#   #   'GEN-588': 't4-tf-2',
#   #   'GEN-581': 't4-tf-3',
#   #   'GEN-582': 't4-tf-4',
#   #   'GEN-583': 't4-tf-5f',
#   # }
#   # labels = {
#   #   'ball3': 'familiar',
#   #   'ball4': 'novel',
#   # }
#   # #
#   # # ids = { #10m
#   # #   'GEN-584': 't4-tf-6',
#   # #   'GEN-585': 't4-tf-7',
#   # #   'GEN-586': 't4-tf-8',
#   # #   'GEN-587': 't4-tf-9',
#   # # }
#   # # labels = {
#   # #   'ball3': 'novel',
#   # #   'ball4': 'familiar',
#   # # }
#   #
#   # ids = { #5m
#   #   'GEN-596': 't4-tf-2',
#   #   'GEN-597': 't4-tf-3',
#   #   'GEN-589': 't4-tf-4',
#   #   'GEN-593': 't4-tf-5f',
#   # }
#   # labels = {
#   #   'ball3': 'familiar',
#   #   'ball4': 'novel',
#   # }
#   #
#   # ids = { #5m
#   #   'GEN-591': 't4-tf-6',
#   #   'GEN-592': 't4-tf-7',
#   #   'GEN-594': 't4-tf-8',
#   #   'GEN-595': 't4-tf-9',
#   # }
#   # labels = {
#   #   'ball3': 'novel',
#   #   'ball4': 'familiar',
#   # }
#   #
#   # ids = { #1m
#   #   'GEN-603': 't4-tf-2',
#   #   'GEN-604': 't4-tf-3',
#   #   'GEN-605': 't4-tf-4',
#   #   'GEN-598': 't4-tf-5f',
#   # }
#   # labels = {
#   #   'ball3': 'familiar',
#   #   'ball4': 'novel',
#   # }
#   #
#   # # ids = { #1m
#   # #   'GEN-599': 't4-tf-6',
#   # #   'GEN-600': 't4-tf-7',
#   # #   'GEN-601': 't4-tf-8',
#   # #   'GEN-602': 't4-tf-9',
#   # #
#   # # }
#   # # labels = {
#   # #   'ball3': 'novel',
#   # #   'ball4': 'familiar',
#   # # }
#   #
#   # ids = { #20m
#   #   'GEN-606': 't4-tf-2',
#   #   'GEN-612': 't4-tf-3',
#   #   'GEN-608': 't4-tf-4',
#   #   'GEN-613': 't4-tf-5f',
#   # }
#   # labels = {
#   #   'ball3': 'familiar',
#   #   'ball4': 'novel',
#   # }
#   #
#   # # ids = { #20m
#   # #   'GEN-609': 't4-tf-6',
#   # #   'GEN-614': 't4-tf-7',
#   # #   'GEN-610': 't4-tf-8',
#   # #   'GEN-611': 't4-tf-9',
#   # #
#   # # }
#   # # labels = {
#   # #   'ball3': 'novel',
#   # #   'ball4': 'familiar',
#   # # }
#   #
#   # ids = { # Play
#   #   'GEN-533': 't4-tf-2f',
#   #   'GEN-534': 't4-tf-3f',
#   #   'GEN-535': 't4-tf-4f',
#   #   'GEN-543': 't4-tf-5',
#   # }
#   # labels = {
#   #   'ball3': 'familiar',
#   #   'ball4': 'novel',
#   # }
#
#   # ids = { # Play
#   #   'GEN-537': 't4-tf-6f',
#   #   'GEN-538': 't4-tf-7f',
#   #   'GEN-539': 't4-tf-8f',
#   #   'GEN-540': 't4-tf-9f',
#   # }
#   # labels = {
#   #   'ball3': 'novel',
#   #   'ball4': 'familiar',
#   # }
#
#   # ids = { # Novel room
#   #   'GEN-619': 't4-tf-6f',
#   # }
#   # labels = {
#   #   'ball3': 'novel',
#   #   'ball4': 'familiar',
#   # }
#
#   #### > BEGIN SEEK_COLOR EXPTS < ####
#
#   # ids = { # seek_yellow
#   #   'GEN-647': 't4-tf-6',
#   #   'GEN-637': 't4-tf-7',
#   #   'GEN-641': 't4-tf-8',
#   #   'GEN-653': 't4-tf-9',
#   # }
#   # labels = {
#   #   'ball3': 'novel',
#   #   'ball4': 'familiar',
#   # }
#   # # ids = { # seek_yellow
#   # #   'GEN-643': 't4-tf-2',
#   # #   'GEN-652': 't4-tf-3',
#   # #   'GEN-645': 't4-tf-4',
#   # #   'GEN-646': 't4-tf-5',
#   # # }
#   # # labels = {
#   # #   'ball3': 'familiar',
#   # #   'ball4': 'novel',
#   # # }
#   #
#   #
#   #
#   # ids = { # seek_magenta
#   #   'GEN-659': 't4-tf-6',
#   #   'GEN-660': 't4-tf-7',
#   #   'GEN-661': 't4-tf-8',
#   #   'GEN-663': 't4-tf-9',
#   # }
#   # labels = {
#   #   'ball3': 'novel',
#   #   'ball4': 'familiar',
#   # }
#   # # ids = { # seek_magenta
#   # #   'GEN-654': 't4-tf-2',
#   # #   'GEN-655': 't4-tf-3',
#   # #   'GEN-656': 't4-tf-4',
#   # #   'GEN-658': 't4-tf-5',
#   # # }
#   # # labels = {
#   # #   'ball3': 'familiar',
#   # #   'ball4': 'novel',
#   # # }
#
#   ids = { # seek_magenta
#     'GEN-887': 't4-tf-2',
#     'GEN-886': 't4-tf-3',
#     'GEN-888': 't4-tf-4',
#     'GEN-889': 't4-tf-5',
#   }
#   labels = {
#     'ball3': 'familiar',
#     'ball4': 'novel',
#   }
#
#   # ids = { # seek_magenta
#   #   'GEN-890': 't4-tf-6',
#   #   'GEN-894': 't4-tf-7',
#   #   'GEN-891': 't4-tf-8',
#   #   'GEN-892': 't4-tf-9',
#   # }
#   # labels = {
#   #   'ball3': 'novel',
#   #   'ball4': 'familiar',
#   # }
#
#
#   ids = { # seek_magenta
#     'GEN-895': 't4-tf-2',
#     'GEN-896': 't4-tf-3',
#     'GEN-897': 't4-tf-4',
#     'GEN-898': 't4-tf-5',
#   }
#   labels = {
#     'ball3': 'familiar',
#     'ball4': 'novel',
#   }
#
#   # ids = { # seek_magenta
#   #   'GEN-899': 't4-tf-6',
#   #   'GEN-900': 't4-tf-7',
#   #   'GEN-901': 't4-tf-8',
#   #   'GEN-902': 't4-tf-9',
#   # }
#   # labels = {
#   #   'ball3': 'novel',
#   #   'ball4': 'familiar',
#   # }
#
#
#   # ids = { # seek_magenta
#   #   'GEN-903': 't4-tf-2',
#   #   'GEN-904': 't4-tf-3',
#   #   'GEN-905': 't4-tf-4',
#   #   'GEN-906': 't4-tf-5',
#   # }
#   # labels = {
#   #   'ball3': 'familiar',
#   #   'ball4': 'novel',
#   # }
#   #
#   # ids = { # seek_magenta
#   #   'GEN-911': 't4-tf-6',
#   #   'GEN-908': 't4-tf-7',
#   #   'GEN-909': 't4-tf-8',
#   #   'GEN-910': 't4-tf-9',
#   # }
#   # labels = {
#   #   'ball3': 'novel',
#   #   'ball4': 'familiar',
#   # }
#
#   ids = { # seek_magenta
#     'GEN-912': 't4-tf-2',
#     'GEN-913': 't4-tf-3',
#     'GEN-914': 't4-tf-4',
#     'GEN-915': 't4-tf-5',
#   }
#   labels = {
#     'ball3': 'familiar',
#     'ball4': 'novel',
#   }
#
#   # ids = { # seek_magenta
#   #   'GEN-916': 't4-tf-6',
#   #   'GEN-917': 't4-tf-7',
#   #   'GEN-918': 't4-tf-8',
#   #   'GEN-919': 't4-tf-9',
#   # }
#   # labels = {
#   #   'ball3': 'novel',
#   #   'ball4': 'familiar',
#   # }
#
#   ids = { # seek_magenta
#     'GEN-948': 't4-tf-2',
#     'GEN-950': 't4-tf-3',
#     'GEN-951': 't4-tf-4',
#     'GEN-952': 't4-tf-5',
#   }
#   labels = {
#     'ball3': 'familiar',
#     'ball4': 'novel',
#   }
#
#   ids = { # seek_magenta
#     'GEN-953': 't4-tf-6',
#     'GEN-954': 't4-tf-7',
#     'GEN-955': 't4-tf-8',
#     'GEN-956': 't4-tf-9',
#   }
#   labels = {
#     'ball3': 'novel',
#     'ball4': 'familiar',
#   }
#
#   ids = { # RND_feat
#     'GEN-994': 't4-tf-2',
#     'GEN-995': 't4-tf-3',
#   }
#   labels = {
#     'ball3': 'familiar',
#     'ball4': 'novel',
#   }
#
#   ids = { # RND_feat
#     'GEN-996': 't4-tf-6',
#     'GEN-997': 't4-tf-7',
#   }
#   labels = {
#     'ball3': 'novel',
#     'ball4': 'familiar',
#   }
#
#   ids = { # RND_feat
#     'GEN-1014': 't4-tf-6',
#   }
#   labels = {
#     'ball3': 'novel',
#     'ball4': 'familiar',
#   }
#
#
#   ids = { # RND_recon (frozen predictor_net)
#     'DRE-15': 't4-tf-8',
#     'DRE-14': 't4-tf-9',
#   }
#   labels = {
#     'ball3': 'novel',
#     'ball4': 'familiar',
#   }
#   colors = {
#             3: 'orange',
#             4: 'magenta',
#             }
#
#   ids = { # RND_recon (frozen predictor_net, various inner loops)
#     'DRE-33': 't4-tf-7',
#     'DRE-31': 't4-tf-8',
#     'DRE-32': 't4-tf-9',
#     'DRE-15': 't4-tf-8',
#   }
#   labels = {
#     'ball3': 'novel',
#     'ball4': 'familiar',
#   }
#
#   ids = { # RND_recon (frozen predictor_net, various inner loops)
#     'DRE-36': 't4-tf-6',
#     'DRE-37': 't4-tf-6',
#     'DRE-40': 't4-tf-8',
#     'DRE-41': 't4-tf-9',
#   }
#   labels = {
#     'ball3': 'novel',
#     'ball4': 'familiar',
#   }
#
#   ids = { # RND_recon (frozen predictor_net, various inner loops)
#     'DRE-42': 't4-tf-2',
#     'DRE-35': 't4-tf-3',
#     'DRE-43': 't4-tf-4',
#     'DRE-39': 't4-tf-5',
#   }
#   labels = {
#     'ball3': 'familiar',
#     'ball4': 'novel',
#   }
#
#   ids = { # RND_recon (frozen predictor_net, various inner loops)
#     'DRE-48': 't4-tf-2',
#     'DRE-49': 't4-tf-3',
#     'DRE-44': 't4-tf-4',
#     'DRE-45': 't4-tf-5',
#   }
#   labels = {
#     'ball3': 'familiar',
#     'ball4': 'novel',
#   }
#
#   ids = { # RND_recon (frozen predictor_net, various inner loops)
#     'DRE-50': 't4-tf-6',
#     'DRE-51': 't4-tf-7',
#     'DRE-46': 't4-tf-8',
#     'DRE-47': 't4-tf-9',
#   }
#   labels = {
#     'ball3': 'novel',
#     'ball4': 'familiar',
#   }
#
#   ids = { # RND_recon (frozen predictor_net, various inner loops)
#     'DRE-56': 't4-tf-6',
#     'DRE-57': 't4-tf-7',
#     'DRE-60': 't4-tf-8',
#     'DRE-61': 't4-tf-9',
#   }
#   labels = {
#     'ball3': 'novel',
#     'ball4': 'familiar',
#   }
#
#   ids = { # RND_recon (frozen predictor_net, various inner loops)
#     'DRE-64': 't4-tf-8',
#     'DRE-65': 't4-tf-9',
#   }
#   labels = {
#     'ball3': 'novel',
#     'ball4': 'familiar',
#   }
#
#   ids = { # RND_recon (frozen predictor_net, various inner loops)
#     'DRE-62': 't4-tf-4',
#     'DRE-63': 't4-tf-5',
#   }
#   labels = {
#     'ball3': 'familiar',
#     'ball4': 'novel',
#   }
#
#   ids = { # RND_recon (rnd_pred_train_every=10)
#     'DRE-73': 't4-tf-2',
#     'DRE-74': 't4-tf-3',
#     'DRE-75': 't4-tf-4',
#     'DRE-76': 't4-tf-5',
#   }
#   labels = {
#     'ball3': 'familiar',
#     'ball4': 'novel',
#   }
#
#   ids = { # RND_recon (rnd_pred_train_every=10)
#     'DRE-77': 't4-tf-6',
#     'DRE-78': 't4-tf-7',
#     'DRE-79': 't4-tf-8',
#     'DRE-80': 't4-tf-9',
#   }
#   labels = {
#     'ball3': 'novel',
#     'ball4': 'familiar',
#   }
#
#   ids = { # p2e (16x w&e, lr 6e-4)
#     'DRE-89': 't4-tf-2',
#     'DRE-90': 't4-tf-3',
#     'DRE-91': 't4-tf-4',
#     'DRE-92': 't4-tf-5',
#   }
#   labels = {
#     'ball3': 'familiar',
#     'ball4': 'novel',
#   }
#
#   ids = { # p2e (16x w&e, lr 6e-4)
#     'DRE-93': 't4-tf-6',
#     'DRE-94': 't4-tf-7',
#     'DRE-95': 't4-tf-8',
#     'DRE-96': 't4-tf-9',
#   }
#   labels = {
#     'ball3': 'novel',
#     'ball4': 'familiar',
#   }
#
#
#   ids = { # p2e (16x w&e, lr 6e-4)
#     'DRE-97': 't4-tf-2',
#     'DRE-98': 't4-tf-3',
#     'DRE-99': 't4-tf-4',
#     'DRE-100': 't4-tf-5',
#   }
#   labels = {
#     'ball3': 'familiar',
#     'ball4': 'novel',
#   }
#
#   ids = { # p2e (16x w&e, lr 6e-4)
#     'DRE-101': 't4-tf-6',
#     'DRE-102': 't4-tf-7',
#     'DRE-103': 't4-tf-8',
#     'DRE-104': 't4-tf-9',
#   }
#   labels = {
#     'ball3': 'novel',
#     'ball4': 'familiar',
#   }
#
#   ids = { # p2e (3balls, no reset_ac)
#     'DRE-105': 't4-tf-2',
#     'DRE-106': 't4-tf-3',
#     'DRE-107': 't4-tf-4',
#     'DRE-108': 't4-tf-5',
#   }
#   labels = {
#     'ball3': 'familiar',
#     'ball4': 'novel',
#     'ball5': 'familiar2'
#   }
#
#   ids = { # p2e (16x w&e)
#     'DRE-113': 't4-tf-2',
#     'DRE-114': 't4-tf-3',
#     'DRE-115': 't4-tf-4',
#     'DRE-116': 't4-tf-5',
#   }
#   labels = {
#     'ball3': 'familiar',
#     'ball4': 'novel',
#   }
#
#   # ids = { # p2e (16x w&e)
#   #   'DRE-117': 't4-tf-6',
#   #   'DRE-118': 't4-tf-7',
#   #   'DRE-119': 't4-tf-8',
#   #   'DRE-120': 't4-tf-9',
#   # }
#   # labels = {
#   #   'ball3': 'novel',
#   #   'ball4': 'familiar',
#   # }
#
#   ids = { # p2e (3balls)
#     'DRE-121': 't4-tf-2',
#     'DRE-122': 't4-tf-3',
#     'DRE-123': 't4-tf-4',
#     'DRE-124': 't4-tf-5',
#   }
#   labels = {
#     'ball3': 'familiar',
#     'ball4': 'novel',
#     'ball5': 'familiar2'
#   }
#
#   ids = { # p2e (3balls)
#     'DRE-125': 't4-tf-6',
#     'DRE-126': 't4-tf-7',
#     'DRE-127': 't4-tf-8',
#     'DRE-128': 't4-tf-9',
#   }
#   labels = {
#     'ball3': 'novel',
#     'ball4': 'familiar',
#     'ball5': 'familiar2'
#   }
#
#   colors = {
#             3: 'orange',
#             4: 'magenta',
#             5: 'red',
#             }
#
#   ids = { # p2e (8x w& 16x e, freeze intr_rew)
#     'DRE-154': 't4-tf-2',
#     'DRE-155': 't4-tf-3',
#     'DRE-156': 't4-tf-4',
#     'DRE-157': 't4-tf-5',
#   }
#   labels = {
#     'ball3': 'familiar',
#     'ball4': 'novel',
#   }
#
#   ids = { # p2e (8x w& 16x e, freeze intr_rew)
#     'DRE-158': 't4-tf-6',
#     'DRE-159': 't4-tf-7',
#     'DRE-160': 't4-tf-8',
#     'DRE-161': 't4-tf-9',
#   }
#   labels = {
#     'ball3': 'novel',
#     'ball4': 'familiar',
#   }
#
#   ids = { # p2e (4x w& 8x e, freeze intr_rew)
#     'DRE-162': 't4-tf-2',
#     'DRE-163': 't4-tf-3',
#     'DRE-164': 't4-tf-4',
#     'DRE-165': 't4-tf-5',
#   }
#   labels = {
#     'ball3': 'familiar',
#     'ball4': 'novel',
#     'ball5': 'familiar2'
#   }
#
#   ids = { # p2e (8x w& 16x e, freeze intr_rew)
#     'DRE-166': 't4-tf-6',
#     'DRE-167': 't4-tf-7',
#     'DRE-168': 't4-tf-8',
#     'DRE-169': 't4-tf-9',
#   }
#   labels = {
#     'ball3': 'novel',
#     'ball4': 'familiar',
#     'ball5': 'familiar2'
#   }
#
#
#   ids = { # p2e nondiscrete
#     'DRE-170': 't4-tf-2',
#     'DRE-171': 't4-tf-3',
#     'DRE-172': 't4-tf-4',
#     'DRE-173': 't4-tf-5',
#   }
#   labels = {
#     'ball3': 'familiar',
#     'ball4': 'novel',
#   }
#   ids = { # p2e nondiscrete
#     'DRE-174': 't4-tf-6',
#     'DRE-175': 't4-tf-7',
#     'DRE-176': 't4-tf-8',
#     'DRE-177': 't4-tf-9',
#   }
#   labels = {
#     'ball3': 'novel',
#     'ball4': 'familiar',
#   }
#

  ############ NONEPISODIC ####################
  ids = { # p2e nondiscrete
    'DRE-271': 't4-tf-3',
    'DRE-275': 't4-tf-2',
  }
  labels = {
    'object1': 'novel',
  }
  colors = {
    'object1': 'orange',
  }

  ids = { # p2e nondiscrete play testing previous working settings
    'DRE-236': 't4-tf-2',
    'DRE-237': 't4-tf-3',
    'DRE-238': 't4-tf-4',
    'DRE-239': 't4-tf-5',
    'DRE-240': 't4-tf-6',
    'DRE-241': 't4-tf-7',
    'DRE-242': 't4-tf-8',
    'DRE-243': 't4-tf-9',
  }
  ids = { # p2e nondiscrete play testing reversion
    'DRE-291': 't4-tf-2',
    'DRE-292': 't4-tf-3',
    # 'DRE-238': 't4-tf-4',
    # 'DRE-239': 't4-tf-5',
  }
  ids = { # p2e nondiscrete nonepisodic
    'DRE-295': 't4-tf-2',
    'DRE-296': 't4-tf-3',
    'DRE-293': 't4-tf-4',
    'DRE-294': 't4-tf-5',
  }
  # ids = { # p2e episode w/ 10k steps
  #   'DRE-297': 't4-tf-4',
  #   'DRE-298': 't4-tf-5',
  # }
  # ids = { # p2e episode w/ 100k steps
  #   'DRE-300': 't4-tf-4',
  #   'DRE-301': 't4-tf-5',
  # }
  # ids = { # p2e nondiscrete nonepisodic reset latent
  #   'DRE-306': 't4-tf-2',
  #   'DRE-307': 't4-tf-3',
  #   'DRE-308': 't4-tf-4',
  # }
  # ids = {
  #   'DRE-309': 't4-tf-5',
  #   'DRE-310': 't4-tf-6',
  #   'DRE-311': 't4-tf-7',
  # }
  # ids = { # p2e nondiscrete nonepisodic reset latent
  #   'DRE-312': 't4-tf-8',
  #   'DRE-313': 't4-tf-9',
  # }

  # ids = { # freerange, reset_state 1k, noise 1.0
  #   'DRE-314': 't4-tf-2',
  #   'DRE-315': 't4-tf-3',
  #   'DRE-316': 't4-tf-4',
  # # }
  #
  # ids = { # freerange, reset_state 1k, noise 10.0
  #   'DRE-317': 't4-tf-5',
  #   'DRE-318': 't4-tf-6',
  #   'DRE-319': 't4-tf-7',
  # }
  #
  # ids = { # freerange, reset_state 1k, reset_position 1k
  #   'DRE-320': 't4-tf-2',
  #   'DRE-321': 't4-tf-3',
  # }
  # ids = { # freerange, reset_position 1k
  #   'DRE-322': 't4-tf-4',
  #   'DRE-323': 't4-tf-5',
  # }
  # # ids = { # freerange, reset_position 10k
  # #   'DRE-324': 't4-tf-6',
  # #   'DRE-325': 't4-tf-7',
  # # }
  # # ids = { # freerange, reset_position 100k
  # #   'DRE-326': 't4-tf-8',
  # #   'DRE-327': 't4-tf-9',
  # # }
  #
  # ids = {  # freerange, reset_position only primary agent, 1k, 10k, 100k
  #   'DRE-328': 't4-tf-2',
  #   'DRE-332': 't4-tf-3',
  #   'DRE-333': 't4-tf-4',
  #   'DRE-334': 't4-tf-5',
  #   'DRE-335': 't4-tf-6',
  #   'DRE-336': 't4-tf-7',
  # }
  #
  # ids = {  # freerange, white
  #   'DRE-338': 't4-tf-2', # 1k reset
  #   'DRE-351': 't4-tf-1', # 1k reset
  #   'DRE-341': 't4-tf-4',
  #   'DRE-342': 't4-tf-5',
  # }

  # ids = {  # freerange, black
  #   'DRE-343': 't4-tf-6', # 1k reset
  #   'DRE-344': 't4-tf-7', # 1k reset
  #   'DRE-346': 't4-tf-8',
  #   'DRE-348': 't4-tf-9',
  # }

  # ids = {  # freerange, white
  #   'DRE-353': 't4-tf-1', # white
  #   'DRE-341': 't4-tf-4', # white
  #   'DRE-342': 't4-tf-5', # white
  # }

  # ids = {  # freerange, black
  #   'DRE-354': 't4-tf-2', # black
  #   'DRE-346': 't4-tf-8', # black
  #   'DRE-348': 't4-tf-9', # black
  #   'DRE-379': 't4-tf-4',  # black
  # }
  # ids = {  # freerange, color
  #   'DRE-355': 't4-tf-4', # color
  #   'DRE-356': 't4-tf-5', # color
  # }
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
  # ids = {  # random policy playground
  #   'DRE-376': 't4-tf-2',
  #   'DRE-375': 't4-tf-1',
  #   'DRE-378': 't4-tf-2',
  #   'DRE-377': 't4-tf-1',
  # }
  ids = {  # labyrinth black
    # 'DRE-381': 't4-tf-1', #10k
    # 'DRE-382': 't4-tf-2',  # 10k
    # 'DRE-390': 't4-tf-5',  # 100k
    # 'DRE-384': 't4-tf-6',  # 100k
    # 'DRE-385': 't4-tf-7',  # freerange
    # 'DRE-386': 't4-tf-8',  # freerange
    'DRE-387': 't4-tf-9',  # random
    'DRE-388': 't4-tf-10',  # random
    'DRE-389': 't4-tf-3',  # random

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
  xlim = int(11000e3 / saverate)

  # xlim = np.arange(0.5e6, 5.5e6, 0.5e6)/saverate
  # xlim = np.array([0.5e6, 1e6, 1.5e6, 2e6, 2.5e6, 3e6])/saverate
  xlim = np.array([2e6, 5e6, 8e6, 11e6])/saverate
  xlim = xlim.astype(int)
  do_plot_trajs = True

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