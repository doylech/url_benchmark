"""
Investigate intrinsic reward, using either logged reward from training,
or by computing intrinsic reward using a checkpointed agent.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime

import elements
import common
from dreamerv2 import agent
import argparse
from tqdm import tqdm

from os.path import expanduser
import training.utils.loading_utils as lu

tf.config.run_functions_eagerly(True)


def compute_intr_reward(agnt, data, do_plot_recon):
  """Compute intrinsic reward, using checkpoint-loaded agent,
  on episodes loaded from replay buffer into a dataset.
  Args:
    agnt - Agent object, loaded from a checkpoint.
    data - dict with keys ['image', 'reward', 'discount', 'action']
    do_plot_recon - bool. Plot a few reconstructed frames to compare with true images.
  Returns:
    ep_rewards - [batch_size x ep_length], intrinsic reward for each step in an episode
  """
  # Loop through timesteps, feeding episodes into model
  nt = data['image'].shape[1]
  state = None
  ep_rewards = []
  for i in tqdm(range(nt)):
    obs = {key: np.expand_dims(val[:, i], axis=1) for key, val in data.items()}

    # Compute latent state features using world model
    if state is None:
      latent = agnt.wm.rssm.initial(len(obs['image']))
      # action = tf.zeros((len(obs['image']), agnt._num_act))
      action = obs['action'][:, 0, :]
      state = latent, action
    else:
      action = obs['action'][:, 0, :]
      state = latent, action
    latent, action = state
    embed = agnt.wm.encoder(agnt.wm.preprocess(obs))
    embed_curr = embed[:, 0, :]
    latent, _ = agnt.wm.rssm.obs_step(latent, action, embed_curr,
                                      sample=False)  # Action, here, is the prev action preceding the current state, I think?
    feat = agnt.wm.rssm.get_feat(latent)  # Concat deter and stoch parts of latent
    # # feat, state, action, disc = agnt.wm.imagine(self.actor, start, hor)

    # Compute intrinsic reward from latent state.
    reward_fn = agnt._expl_behavior._intr_reward
    rewards = reward_fn(feat, state, action)
    intr_rewards = rewards[1]
    ep_rewards.append(intr_rewards.numpy())

    # Try decoding state to an image to see if it matches.
    if do_plot_recon:
      if i == 200: # Pick an arbitrary timepoint in the episode
        recon = agnt.wm.heads['image'](feat).mode().numpy()
        true = data['image'][:, i + 1]
        plt.figure()
        for ind in range(0, min(6, true.shape[0]), 2):
          plt.subplot(3, 2, ind + 1)
          plt.imshow(true[ind, :] / 255.)
          plt.axis('off')
          if ind == 0:
            plt.title('True')
          plt.subplot(3, 2, ind + 2)
          plt.imshow(recon[ind, :] - np.min(recon[0, :]))
          plt.axis('off')
          if ind == 0:
            plt.title('Recon')
        plt.tight_layout()
        plt.show()

  ep_rewards = np.vstack(ep_rewards).T
  return ep_rewards


def get_log_with_intr_reward(basedir, intr_fname, log_fname, do_plot=True):
  """Load csv files with intrinsic reward, and with exploration logs, and
  merge them, using the 'total_step' counter.
  Args:
    basedir: i.e. /home/user/logs/expid/
    intr_name: i.e. intr_reward_ee_env-0.csv
    log_name: i.e.  log_train_env0.csv

  Returns:
    df: pandas dataframe merging intrinsic reward with exploration logs.
  """
  fn_intr = f'{basedir}/{intr_fname}'
  action_repeat = 2 # This is the config for dmc environments.
  df_intr = pd.read_csv(fn_intr)
  df_intr['total_step'] = df_intr['step']*action_repeat # Env logging goes 'action_repeat' steps for every 1 train step
  df_intr = df_intr.drop(columns=['step'])

  fn_log = f'{basedir}/{log_fname}'
  df_log = pd.read_csv(fn_log)

  df = pd.merge(df_intr, df_log, on='total_step')

  if do_plot:
    plt.figure() # To compare with tensorboard plot of explor_reward_intr
    plt.plot(df['total_step'], df['expl_reward_intr_mean'])
    plt.xlabel('Total step')
    plt.ylabel('expl_reward_intr_mean')
    plt.title(f'{intr_fname}, {log_fname}')

    plt.figure() # All intr reward, by location
    plt.scatter(df['agent0_xloc'], df['agent0_yloc'], c=df['expl_reward_intr_mean'],
                alpha=0.2, s=2, cmap='coolwarm')
    plt.xlim([-20, 20]), plt.ylim([-20, 20]), plt.colorbar()
    plt.title(f'Logged: {expid}, {buffer_name}')
    plt.show()

  return df


def plot_curious_frames(all_ep_rewards, basedir, buffer_name, n_to_plot=10):
  """
  Plot frames with highest intrinsic reward.
  Plot one frame per episode.
  Args:
    all_ep_rewards - [n_episodes x n_timepoints]
    basedir - directory of exp
    buffer_name - replay buffer directory name
    n_to_plot - number of episodes to plot
  """
  plt.figure(figsize=(3, 1.5*int(n_to_plot/2)))
  max_inds = np.argmax(all_ep_rewards, axis=1)
  max_vals = np.max(all_ep_rewards, axis=1)
  sort_eps = np.flipud(np.argsort(max_vals))
  for iter, which_ep in enumerate(sort_eps[:n_to_plot]):
    ind = max_inds[which_ep]
    reward_val = all_ep_rewards[which_ep, ind]
    ep, name, timestamp = lu.load_ep(basedir, which_ep=which_ep, buffer_name=buffer_name)
    image = ep['image'][ind, :]
    plt.subplot(int(n_to_plot/2), 2, iter+1)
    plt.imshow(image)
    plt.axis('off')
    plt.title(f'Ep:{which_ep}, Ind:{ind}\n Rew:{reward_val:.4f}')
  plt.tight_layout()


if __name__ == "__main__":
  ## Set options for script.
  expid = 'GEN-308'
  env_num = 0
  buffer_name = f'train_replay_{env_num}'
  #buffer_name = 'test_env_train_replay_1-1'
  # checkpoint_name = 'variables_train_agent_envindex0.pkl'
  checkpoint_name = 'variables_train_agent.pkl'

  home = expanduser("~")
  basedir = f"{home}/logs/{expid}"
  # plot_lims=[-15,15]
  plot_lims=[-20,20]

  # TODO: Plot GEN-312 and GEN-315 (after downloading eps from remote)

  ## Load agent from checkpoint
  agnt = lu.load_agent(basedir, env_num, batch_size=5,
                       checkpoint_name=checkpoint_name)

  do_compute_intr_reward = True
  do_compare_with_log_csv = False
  do_plot_curious_frames = True

  if do_compute_intr_reward:
    batch_size = 100
    n_batches = 1
    n_batches = int(len(os.listdir(f"{basedir}/{buffer_name}/"))/batch_size)

    all_xyz = []
    all_ep_rewards = []
    for which_batch in range(n_batches):
      eps = lu.load_eps(basedir, buffer_name, batch_size, start_ind=which_batch*batch_size)
      xyz = np.stack([ep['absolute_position_agent0'] for ep in eps], axis=0)
      data = {key: np.vstack([np.expand_dims(ep[key].astype('float32'), axis=0) for ep in eps])
              for key in ['image', 'reward', 'discount', 'action']}
      ep_rewards = compute_intr_reward(agnt, data, do_plot_recon=True)

      all_xyz.append(xyz)
      all_ep_rewards.append(ep_rewards)

    all_xyz = np.vstack(all_xyz)
    all_ep_rewards = np.vstack(all_ep_rewards)
    flat_xyz = all_xyz.reshape(all_xyz.shape[0]*all_xyz.shape[1], -1)
    flat_ep_rewards = all_ep_rewards.reshape(all_ep_rewards.shape[0]*all_ep_rewards.shape[1], -1)

    plt.figure()
    plt.scatter(flat_xyz[:, 0], flat_xyz[:, 1], c=flat_ep_rewards, alpha=0.5, s=0.2, cmap='coolwarm')
    plt.xlim(plot_lims), plt.ylim(plot_lims), plt.colorbar()
    plt.title(f'{expid},{buffer_name}\n{checkpoint_name}')
    plt.show()
    print('done')

    plt.figure()
    plt.plot(flat_ep_rewards)
    plt.xlabel('step'), plt.ylabel('intr reward')
    plt.title(f'{expid},{buffer_name}\n{checkpoint_name}')
    plt.show()

    if do_plot_curious_frames: # Plot frames with highest intr_reward
      plot_curious_frames(all_ep_rewards, basedir, buffer_name, n_to_plot=10)
      plt.suptitle(f'{expid},{buffer_name}\n{checkpoint_name}')
      plt.tight_layout()
      plt.show()

  if do_compare_with_log_csv:  # Compare computed intr_reward with saved out intr_reward
    intr_fname = f'intr_reward_ee_env-{env_num}.csv'
    log_fname = f'log_train_env{env_num}.csv'
    df = get_log_with_intr_reward(basedir, intr_fname=intr_fname, log_fname=log_fname)

    # Now, load an episode, and compute intr_reward using agent
    ep, name, timestamp = lu.load_ep(basedir, which_ep=-2, buffer_name=buffer_name)
    eps = [ep]
    data = {key: np.vstack([np.expand_dims(ep[key].astype('float32'), axis=0) for ep in eps])
            for key in ['image', 'reward', 'discount', 'action']}
    ep_rewards = compute_intr_reward(agnt, data, do_plot_recon=True)
    xyz = ep['absolute_position_agent0']

    # Get log entries corresponding to that episode
    log_timestamp = lu.get_closest_timestamp(df, timestamp)
    df_ep = df[df['timestamp'] == log_timestamp]

    # Plot comparison of logged and computed intr_reward
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)  # Plot computed intr_reward
    plt.scatter(xyz[:, 0], xyz[:, 1], c=ep_rewards, alpha=0.5, s=0.2, cmap='coolwarm')
    plt.xlim(plot_lims), plt.ylim(plot_lims), plt.colorbar()
    plt.title(f'Computed: {expid},{buffer_name}\n{checkpoint_name}')

    plt.subplot(1, 2, 2)  # Plot logged intr_reward
    plt.scatter(df_ep['agent0_xloc'], df_ep['agent0_yloc'], c=df_ep['expl_reward_intr_mean'],
                alpha=0.5, s=2, cmap='coolwarm')
    plt.xlim(plot_lims), plt.ylim(plot_lims), plt.colorbar()
    plt.title(f'Logged: {expid}, {buffer_name}, \n{log_timestamp}')
    plt.show()

  print('done')

