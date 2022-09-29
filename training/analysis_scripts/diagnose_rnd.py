"""
Investigate rnd intrinsic reward, using either logged reward from training,
or by computing intrinsic reward using a checkpointed agent.
Then use analyze_intrinsic_reward.py to plot results.
Note: this is a fork of diagnose_intrinsic_reward.py
"""
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

tf.config.run_functions_eagerly(True)


if __name__ == "__main__":
  ## Set options for script.
  expid = 'GEN-994'
  bufferid = 'GEN-EXAMPLE_EPS'
  # bufferid = expid
  # replay_buffer_name = 'test_env_train_replay_1-1'
  checkpoint_name = 'variables_train_agent_envindex0_000102500.pkl'
  # checkpoint_name = 'variables_train_agent_envindex0_000502500.pkl'
  # checkpoint_name = 'variables_train_agent_envindex0_000052500.pkl'
  # checkpoint_name = 'variables_train_agent_envindex0_000102500.pkl'
  # checkpoint_name = 'variables_train_agent_envindex0_000062500.pkl'
  replay_buffer_name = 'train_replay_0'

  home = expanduser("~")
  basedir = f"{home}/logs/{expid}"
  bufferdir = f"{home}/logs/{bufferid}"
  # plot_lims=[-15,15]
  plot_lims=[-20,20]

  plot_dir = f'{basedir}/plots'
  os.makedirs(plot_dir, exist_ok=True)

  from tensorflow.keras.mixed_precision import experimental as prec
  prec.set_policy(prec.Policy('mixed_float16'))

  ## Load agent from checkpoint
  agnt = lu.load_agent(basedir,
                       checkpoint_name=checkpoint_name,
                       batch_size=5,
                       deterministic=False
                       )

  do_compute_intr_reward = True
  do_plot_curious_frames = True
  do_compare_with_log_csv = False

  if do_compute_intr_reward:
    batch_size = 2
    skip_size = 1
    which_eps = np.arange(0, len(os.listdir(f"{bufferdir}/{replay_buffer_name}/")), skip_size)
    # which_eps = np.arange(0, 504, skip_size)
    # which_eps = np.arange(505, len(os.listdir(f"{basedir}/{replay_buffer_name}/")), skip_size)

    # which_eps = np.arange(0, 15)
    n_batches = int(np.ceil(len(which_eps)/batch_size))
    print(f'Total batches{n_batches}')

    discounts = [0, 0.99]
    all_xyz = []
    all_ep_rewards = {d: [] for d in discounts}
    all_image_likes = []
    all_reward_likes = []
    all_kl = []

    t0 = time.time()
    # for which_batch in range(start_batch, start_batch+n_batches):
    for which_batch in range(n_batches):

      print(f'Batch: {which_batch}/{n_batches}')
      batch_eps = which_eps[which_batch*batch_size:which_batch*batch_size+batch_size]

      # TODO: Try this with Replay as well?
      # nt = agnt.config.dataset.length
      nt = 501
      eps = lu.load_eps(bufferdir, replay_buffer_name, batch_eps=batch_eps)

      xyz = np.stack([ep['absolute_position_agent0'][:nt,:] for ep in eps], axis=0)
      data = {key: np.vstack([np.expand_dims(ep[key].astype('float32'), axis=0)
                              for ep in eps])[:,:nt]
              for key in eps[0].keys()}
      data = {key: tf.cast(data[key], dtype=data[key].dtype) for key in data.keys()}

      ep_rewards, outs, model_loss, metrics = du.compute_intr_reward(agnt, data,
                                                                  discounts=discounts,
                                                                  deterministic=True,
                                                                  do_plot_recon=False)

      all_xyz.append(xyz)
      [all_ep_rewards[d].append(ep_rewards[d]) for d in discounts]

      all_image_likes.append(outs['likes']['image'])
      all_reward_likes.append(outs['likes']['reward'])
      all_kl.append(outs['kl'])

    all_xyz_v = np.vstack(all_xyz)
    flat_xyz = all_xyz_v.reshape(all_xyz_v.shape[0]*all_xyz_v.shape[1], -1)

    all_image_likes_v = np.vstack(all_image_likes)
    all_reward_likes_v = np.vstack(all_reward_likes)
    all_kl_v = np.vstack(all_kl)

    flat_image_likes = all_image_likes_v.reshape(all_image_likes_v.shape[0]*all_image_likes_v.shape[1], -1)
    flat_reward_likes = all_reward_likes_v.reshape(all_reward_likes_v.shape[0]*all_reward_likes_v.shape[1], -1)
    flat_kl = all_kl_v.reshape(all_kl_v.shape[0]*all_kl_v.shape[1], -1)

    all_ep_rewards_v = {}
    flat_ep_rewards = {}
    for d in discounts:
      all_ep_rewards_v[d] = np.vstack(all_ep_rewards[d])
      shape = all_ep_rewards_v[d].shape
      flat_ep_rewards[d] = all_ep_rewards_v[d].reshape(shape[0]*shape[1], -1)

    # Save results for later plotting
    fname = f'{basedir}/intr_rew_{replay_buffer_name}_nt{nt}__{checkpoint_name}__{batch_size}_{n_batches}_{skip_size}.pkl'
    with open(fname, 'wb') as f:
      ddd = {'all_xyz_v':all_xyz_v,'all_ep_rewards_v':all_ep_rewards_v,'discounts': discounts,
             'replay_buffer_name':replay_buffer_name, 'checkpoint_name': checkpoint_name, 'nt':nt,
             'batch_size': batch_size, 'n_batches':n_batches, 'which_eps': which_eps, #'start_batch': start_batch,
             'all_image_likes_v': all_image_likes_v, 'all_reward_likes_v': all_reward_likes_v,
             'all_kl_v': all_kl_v}
      pickle.dump(ddd, f)

    # Make plots
    info = {
      'expid': expid,
      'replay_buffer_name': replay_buffer_name,
      'checkpoint_name': checkpoint_name,
      'batch_size': batch_size,
      'skip_size': skip_size,
      'n_batches': n_batches,
      'nt': nt,
      'plot_lims': plot_lims,
      'basedir': basedir,
      'bufferdir': bufferdir,
    }

    du.make_plots(flat_xyz, flat_image_likes, flat_reward_likes, flat_kl, flat_ep_rewards,
                  discounts, info, plot_dir, all_ep_rewards_v)


  if do_compare_with_log_csv:  # Compare computed intr_reward with saved out intr_reward
    env_num = 0
    intr_fname = f'intr_reward_ee_env-{env_num}.csv'
    log_fname = f'log_train_env{env_num}.csv'
    df = du.get_log_with_intr_reward(basedir, intr_fname=intr_fname, log_fname=log_fname, expid=expid)

    # Now, load an episode, and compute intr_reward using agent
    ep, name, timestamp = lu.load_ep(basedir, which_ep=5, buffer_name=replay_buffer_name)
    eps = [ep]
    data = {key: np.vstack([np.expand_dims(ep[key].astype('float32'), axis=0) for ep in eps])
            for key in ['image', 'reward', 'discount', 'action']}
    out = du.compute_intr_reward(agnt, data, do_plot_recon=True)
    ep_rewards = out[0][0]
    xyz = ep['absolute_position_agent0']

    # Get log entries corresponding to that episode
    log_timestamp = lu.get_closest_timestamp(df, timestamp)
    df_ep = df[df['timestamp'] == log_timestamp]

    # Plot comparison of logged and computed intr_reward
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)  # Plot computed intr_reward
    plt.scatter(xyz[:, 0], xyz[:, 1], c=ep_rewards, alpha=0.5, s=0.2, cmap='coolwarm')
    plt.xlim(plot_lims), plt.ylim(plot_lims), plt.colorbar()
    plt.title(f'Computed: {expid},{replay_buffer_name}\n{checkpoint_name}')

    plt.subplot(1, 2, 2)  # Plot logged intr_reward
    plt.scatter(df_ep['agent0_xloc'], df_ep['agent0_yloc'], c=df_ep['expl_reward_intr_mean'],
                alpha=0.5, s=2, cmap='coolwarm')
    plt.xlim(plot_lims), plt.ylim(plot_lims), plt.colorbar()
    plt.title(f'Logged: {expid}, {replay_buffer_name}, \n{log_timestamp}')
    plt.show()

  print('done')

