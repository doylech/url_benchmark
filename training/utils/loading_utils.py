"""
Utilities for loading agent from checkpoint and associated per-episode logging.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip
import pickle
import pathlib
import shutil
import tensorflow as tf
from datetime import datetime

import elements
import common
import common.envs
from dreamerv2 import agent
import argparse
from tqdm import tqdm

from os.path import expanduser

tf.config.run_functions_eagerly(True)


def get_closest_timestamp(df, traj_timestamp):
  a = datetime.strptime(traj_timestamp, '%Y%m%dT%H%M%S')
  min_t = None
  min_delta = np.inf
  for t in df['timestamp'].unique():
    b = datetime.strptime(t, '%Y%m%dT%H%M%S')
    delta = b - a
    delta = delta.seconds
    if np.abs(delta) < min_delta:
      if delta >= 0:
        min_delta = delta
        min_t = t

  return min_t


def load_ep(basedir, which_ep, buffer_name='train_replay_0', timestamp=None):
  """Load an episode from replay buffer"""
  # print(f'{basedir}, Ep: {which_ep}')
  traj_dir = f"{basedir}/{buffer_name}/"
  traj_names = os.listdir(traj_dir)
  traj_names.sort()
  if timestamp is None:
    traj_name = traj_names[which_ep]
  else:
    traj_name = [x for x in traj_names if timestamp in x]
    traj_name = traj_name[0]
  print(traj_name)
  traj_timestamp = traj_name.split('-')[0]
  traj_path = f'{traj_dir}/{traj_name}'
  ep = np.load(traj_path)
  return ep, traj_name, traj_timestamp


def load_eps(basedir, buffer_name, batch_size=None, start_ind=0, step=1, batch_eps=None):
  """Load multiple episodes into a batch"""
  # print(f'Loading episodes from {basedir}')
  eps = []
  # for which_ep in tqdm(range(start_ind, start_ind + batch_size, step)):
  if batch_eps is None:
    batch_eps = np.arange(start_ind, start_ind + batch_size, step)
  for which_ep in batch_eps:
    print(which_ep, end='...')
    ep, name, timestamp = load_ep(basedir, which_ep=which_ep, buffer_name=buffer_name)
    eps.append(ep)
  return eps


def load_agent(basedir, checkpoint_name='variables_train_agent.pkl',
               batch_size=None, deterministic=False, do_load_ckpt=True):
  """Load a checkpointed Dreamer agent.
  args:
    basedir: str. i.e. /home/saal2/logs/GEN-304
    env_num: int. which environment number agent corresponds to (i.e. from multi-environment playgrounds)
  """

  # First load/initialize all the various stuff needed to init an agent
  for_ckpt_dir = f'{basedir}/for_ckpt'
  logdir = f'{basedir}/fiddle_dir'
  os.makedirs(logdir, exist_ok=True)

  with open(f'{for_ckpt_dir}/config_{checkpoint_name}', 'rb') as f:
    config = pickle.load(f)

  # Set precision. This is important!
  if config.precision == 16:
    from tensorflow.keras.mixed_precision import experimental as prec
    prec.set_policy(prec.Policy('mixed_float16'))
    print('Setting precision to mixed_float16')
  elif config.precision == 32:
    from tensorflow.keras.mixed_precision import experimental as prec
    prec.set_policy(prec.Policy('float32'))
    print('Setting precision to float32')
  from tensorflow.keras import mixed_precision as prec
  print(prec.global_policy().compute_dtype)


  if batch_size is not None:
    config = config.update({'dataset.batch': batch_size})

  with open(f'{for_ckpt_dir}/action_space_{checkpoint_name}', 'rb') as f:
    action_space = pickle.load(f)

  step = elements.Counter(0)
  outputs = [
      elements.TerminalOutput(),
      elements.JSONLOutput(logdir),
      elements.TensorBoardOutput(logdir),
  ]
  logger = elements.Logger(step, outputs, multiplier=config.action_repeat)

  # Copy a few episodes over from main replay buffer so can init dataset
  env_num = 0
  os.makedirs(os.path.join(logdir, f'train_replay_{env_num}'), exist_ok=True)
  traj_dir = f"{basedir}/train_replay_{env_num}/"
  eps = os.listdir(traj_dir)
  eps.sort()
  for traj_name in eps[:2]:
    shutil.copy(os.path.join(traj_dir, traj_name), os.path.join(logdir, f'train_replay_{env_num}', traj_name))

  replay_dir = pathlib.Path(f'{logdir}/train_replay_{env_num}')
  # os.makedirs(replay_dir, exist_ok=True)
  train_replay = common.Replay(replay_dir, config.replay_size, config)
  print('Num episodes in train_replay: ', train_replay.num_episodes)
  # if train_replay.num_episodes == 0:
  #   random_agent = common.RandomAgent(action_space)
  #   train_envs = [common.envs.make_env(config, env_sequence_item.name, train_logging_params=None) for _ in range(config.num_envs)]
  #   train_driver = common.Driver(train_envs, logdir=logdir)
  #   train_driver(random_agent, episodes=1)
  #   train_driver.reset()
    # Copy over to eval_replay_{env_num}

  train_dataset = iter(train_replay.dataset(**config.dataset, deterministic=deterministic))

  print(f'--->Beginning agent initialization!')
  agnt = agent.Agent(config, logger, action_space, step, train_dataset)
  print(f'--->Agent initialized!')
  if do_load_ckpt:
    agnt.load(f'{basedir}/{checkpoint_name}')
    print(f'--->Agent loaded from checkpoint! {checkpoint_name}')
  return agnt