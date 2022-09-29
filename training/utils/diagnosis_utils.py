"""
Investigate intrinsic reward, using either logged reward from training,
or by computing intrinsic reward using a checkpointed agent.
"""
import os
import sys
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

import scipy.stats

from os.path import expanduser
import training.utils.loading_utils as lu

tf.config.run_functions_eagerly(True)

def summarize_quadrants(flat_xyz, z_img_loss, z_kl, z_ep_rewards, w=1):
  """Summary statistic within each quadrant, for each metric.
  Args:
    w - halfwidth of center box to exclude
  """
  # Quantify quadrant
  discounts = list(z_ep_rewards.keys())
  quad = {}
  quad[0] = np.where((flat_xyz[:, 0] < -w) & (flat_xyz[:, 1] < -w))[0]
  quad[1] = np.where((flat_xyz[:, 0] > w) & (flat_xyz[:, 1] < -w))[0]
  quad[2] = np.where((flat_xyz[:, 0] > w) & (flat_xyz[:, 1] > w))[0]
  quad[3] = np.where((flat_xyz[:, 0] < -w) & (flat_xyz[:, 1] > w))[0]
  qlabels = {0: 'bl', 1: 'br', 2: 'ur', 3: 'ul'}
  img_quad = {}; kl_quad = {}; intr_rew_quad = {d:{} for d in discounts}
  for i in np.arange(4):
    img_quad[i] = np.mean(z_img_loss[quad[i]])
    kl_quad[i] = np.mean(z_kl[quad[i]])
    for d in discounts:
      intr_rew_quad[d][i] = np.mean(z_ep_rewards[d][quad[i]])
  # TODO: Make a pandas array?

  plt.figure()
  plt.subplot(1,4,1)
  x = np.arange(4)
  plt.bar(x, [img_quad[i] for i in x])
  plt.xticks(x, [qlabels[i] for i in x])
  plt.title('Img loss')

  plt.subplot(1,4,2)
  x = np.arange(4)
  plt.bar(x, [kl_quad[i] for i in x])
  plt.xticks(x, [qlabels[i] for i in x])
  plt.title('KL')

  for di in [0, 1]:
    plt.subplot(1,4,3+di)
    x = np.arange(4)
    plt.bar(x, [intr_rew_quad[discounts[di]][i] for i in x])
    plt.xticks(x, [qlabels[i] for i in x])
    plt.title(f'Intr rew {discounts[di]}')


def make_plots(flat_xyz, flat_image_likes, flat_reward_likes, flat_kl,  flat_ep_rewards,
               discounts, info, plot_dir, all_ep_rewards_v=None):
  """Make summary plots with output from diagnose_intrinsic_reward.py"""

  expid = info['expid']
  replay_buffer_name = info['replay_buffer_name']
  checkpoint_name = info['checkpoint_name']
  plot_lims = info['plot_lims']
  basedir = info['basedir']
  bufferdir = info['bufferdir']

  # Compute z-score
  flat_image_loss = -flat_image_likes
  z_img_loss = scipy.stats.zscore(flat_image_loss) #(flat_image_loss - np.mean(flat_image_loss)) / np.std(flat_image_loss)
  z_kl = scipy.stats.zscore(flat_kl)
  z_ep_rewards = {}
  for d in discounts:
    z_ep_rewards[d] = scipy.stats.zscore(flat_ep_rewards[d])

  # Plot summary over quadrants
  summarize_quadrants(flat_xyz, z_img_loss, z_kl, z_ep_rewards, w=1)
  plt.suptitle(f'Quadrants {expid},{replay_buffer_name}\n{checkpoint_name}')
  plt.tight_layout()
  plt.show()

  suffix = f'{expid}_{replay_buffer_name}_{checkpoint_name}'
  # Plot visitation count
  plt.figure(),
  plt.plot(flat_xyz[:, 0], flat_xyz[:, 1], '.', markersize=0.1, alpha=0.05),
  plt.title(f'{expid},{replay_buffer_name}\nVisitation across plotted episodes')
  plt.savefig(f'{plot_dir}/visitation_{suffix}.png')
  plt.show()

  # Plot model loss
  plt.figure()
  flat_image_loss = -flat_image_likes
  image_clim = [np.percentile(flat_image_loss, 1), np.percentile(flat_image_loss, 99)]
  plt.scatter(flat_xyz[:, 0], flat_xyz[:, 1], c=flat_image_loss,
              # alpha=0.5,
              alpha=np.clip(10 * flat_image_loss, 0.0, 1.0),
              s=0.2, cmap='coolwarm',
              )
  plt.clim(image_clim)
  plt.xlim(plot_lims), plt.ylim(plot_lims), \
  # plt.colorbar()
  plt.title(f'Img loss, {expid},{replay_buffer_name}\n{checkpoint_name}')
  plt.savefig(f'{plot_dir}/img_loss_{suffix}.png')
  plt.show()
  print('done')

  # Plot kl loss
  kl_clim = [np.percentile(flat_kl, 1), np.percentile(flat_kl, 99)]
  plt.figure()
  plt.scatter(flat_xyz[:, 0], flat_xyz[:, 1], c=flat_kl,
              alpha=0.5, s=0.2, cmap='coolwarm')
  plt.clim(kl_clim)
  plt.xlim(plot_lims), plt.ylim(plot_lims), \
  plt.colorbar()
  plt.title(f'KL loss, {expid},{replay_buffer_name}\n{checkpoint_name}')
  plt.savefig(f'{plot_dir}/kl_loss_{suffix}.png')
  plt.show()


  for d in discounts:
    plt.figure()
    plt.scatter(flat_xyz[:, 0], flat_xyz[:, 1], c=flat_ep_rewards[d], alpha=0.5, s=0.2, cmap='coolwarm')
    plt.xlim(plot_lims), plt.ylim(plot_lims),
    plt.colorbar()
    if d == 0:
      # plt.clim(kl_clim)
      # plt.clim(image_clim)
      # plt.clim([0, 1])
      pass
    plt.title(f'{expid},{replay_buffer_name}\n{checkpoint_name}, d{d}')
    plt.savefig(f'{plot_dir}/intr_reward_{suffix}_d{d}.png')
    plt.show()

    if all_ep_rewards_v is not None:
      # Plot frames with highest intr_reward
      plot_curious_frames(all_ep_rewards_v[d], bufferdir, replay_buffer_name, n_to_plot=10)
      plt.suptitle(f'{expid},{replay_buffer_name}_d{d}\n{checkpoint_name}', fontsize=7)
      plt.tight_layout()
      plt.savefig(f'{plot_dir}/curious_frames_{suffix}_d{d}.png')
      plt.show()


def get_log_with_intr_reward(basedir, intr_fname, log_fname, expid, do_plot=True,):
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
    plt.title(f'Logged: {expid}')
    plt.show()

  return df


def plot_curious_frames(all_ep_rewards, bufferdir, buffer_name, n_to_plot=10):
  """
  Plot frames with highest intrinsic reward.
  Plot one frame per episode.
  Args:
    all_ep_rewards - [n_episodes x n_timepoints]
    bufferdir - directory of exp
    buffer_name - replay buffer directory name
    n_to_plot - number of episodes to plot
  """
  plt.figure(figsize=(3, 1.5*int(n_to_plot/2)))
  max_inds = np.argmax(all_ep_rewards, axis=1)
  max_vals = np.max(all_ep_rewards, axis=1)
  sort_eps = np.flipud(np.argsort(max_vals))
  delta = 10
  for iter, which_ep in enumerate(sort_eps[:n_to_plot]):
    ind = max_inds[which_ep]
    reward_val = all_ep_rewards[which_ep, ind+delta]
    ep, name, timestamp = lu.load_ep(bufferdir, which_ep=which_ep, buffer_name=buffer_name)
    image = ep['image'][ind+delta, :]
    plt.subplot(int(n_to_plot/2), 2, iter+1)
    plt.imshow(image)
    plt.axis('off')
    plt.title(f'Ep:{which_ep}, Ind:{ind+delta}\n Rew:{reward_val:.4f}')
  plt.tight_layout()
  plt.show()

def get_discounted_rewards(intr_reward, discounts, shape=None):
  """ 
  Compute discounted reward across imagined horizon.
  :param intr_reward: [horizon_t, n_timepoints]
  :param discounts: list of floats
  :return: dict with discounted reward for each discount
  """
  discounted_rewards = {}
  for discount in discounts:
    if discount == 0:
      summary_reward = intr_reward[0, :]
    else:
      scaled = intr_reward.copy()
      for i in range(scaled.shape[0]):
        scaled[i, :] = scaled[i, :] * (discount ** i)
      summary_reward = np.sum(scaled, axis=0)

    if shape is not None:
      summary_reward = summary_reward.reshape(shape)
    discounted_rewards[discount] = summary_reward
  return discounted_rewards

def compute_intr_reward(agnt, data, discounts=[0], deterministic=True,  do_plot_recon=False):
  """Compute intrinsic reward, using checkpoint-loaded agent,
  on episodes loaded from replay buffer into a dataset.
  Args:
    agnt - Agent object, loaded from a checkpoint.
    data - dict with keys ['image', 'reward', 'discount', 'action']
    discount - if 0, then only return reward of first time step.
               if >0, then include the later timesteps in the imagined sequence.
    do_plot_recon - bool. Plot a few reconstructed frames to compare with true images.
  Returns:
    ep_rewards - [batch_size x ep_length], intrinsic reward for each step in an episode
  """
  # computed_preproc = agnt.wm.preprocess(data)
  # computed_embed = agnt.wm.encoder(computed_preproc)
  # post, prior = agnt.wm.rssm.observe(computed_embed,
  #                                    computed_preproc['action'],
  #                                    state=None,
  #                                    sample=(not deterministic))

  model_loss, post, outs, metrics =  agnt.wm.loss(data, state=None, sample=(not deterministic))

  start = post

  (feat, state, ## state is: for t in horizon; for each starting latent state (from each timepoint of each batch); the latent state
   action, disc) = agnt.wm.imagine(agnt._expl_behavior.ac.actor,
                                   start,
                                   agnt.config.imag_horizon,
                                   deterministic=deterministic)
  rewards = agnt._expl_behavior._intr_reward(feat, state, action)
  intr_reward = rewards[1]
  intr_reward = intr_reward.numpy()

  if do_plot_recon:
    for i in range(0, 500, 40):
      plt.figure(), plt.imshow(data['image'][0, i, :, :, :] / 255.), plt.title(i), plt.show()

    # recon = agnt.wm.heads['image'](feat[0, :10, :]).mode().numpy().astype(np.float32)
    # true = data['image'][0, :10, :, :, :]

    ## For gen-446
    # recon = agnt.wm.heads['image'](feat[0, 501+205:501+215, :]).mode().numpy().astype(np.float32)
    # true = data['image'][1, 205:215, :, :, :]
    # recon = agnt.wm.heads['image'](feat[0, 501+35:501+45, :]).mode().numpy().astype(np.float32)
    # true = data['image'][1, 35:45, :, :, :]

    # recon = agnt.wm.heads['image'](feat[0, 355:365, :]).mode().numpy().astype(np.float32)
    # true = data['image'][0, 355:365, :, :, :]
    recon = agnt.wm.heads['image'](feat[0, 475:485, :]).mode().numpy().astype(np.float32)
    true = data['image'][0, 475:485, :, :, :]


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


    # if i == 200:  # Pick an arbitrary timepoint in the episode
    #   recon = agnt.wm.heads['image'](feat).mode().numpy()
    #   true = data['image'][:, i + 1]
    #   plt.figure()
    #   for ind in range(0, min(6, true.shape[0]), 2):
    #     plt.subplot(3, 2, ind + 1)
    #     plt.imshow(true[ind, :] / 255.)
    #     plt.axis('off')
    #     if ind == 0:
    #       plt.title('True')
    #     plt.subplot(3, 2, ind + 2)
    #     plt.imshow(recon[ind, :] - np.min(recon[0, :]))
    #     plt.axis('off')
    #     if ind == 0:
    #       plt.title('Recon')
    #   plt.tight_layout()
    #   plt.show()

  discounted_rewards = get_discounted_rewards(intr_reward, discounts,
                                              shape=post['deter'].shape[:2])

  return discounted_rewards, outs, model_loss, metrics



def dream(agnt, data, actions_for_imagine, save_dir,
          expid='',
          start_imagining_at=5, burnin=5, imagine_for=20,
          deterministic_wm=True,
          show_burnin_recon=False,
          include_truth_images=True,
          plot_image_loss=True,
          plot_intrinsic_reward=True,
          do_save_gif=True,
          deterministic_expl=True,
          discounts=[0, 0.99],
          ylim_imgloss=None,
          ylim_intr_rew=None,
          key_t=None,
          do_return_imgs=False,
          do_include_error_img=False,
          fig_suffix='png',
          no_plots=False):
    """
    Do a rollout, visualize what is being dreamt, compare with the true data,
    and save out a movie comparing the two.

    Args:
      agnt - Agent object, loaded from a checkpoint.
      data - dict with keys ['image', 'reward', 'discount', 'action']
      actions_for_imagine: [imagine_for x action_size]
      save_dir: location to save video output
      start_imagining_at: timestep in episode to begin openloop imagination
      burnin: how many timesteps in episode to observe before imagination
      imagine_for: how many imagined timesteps
      deterministic_wm: whether to sample latent state or use mode()
      show_burnin_recon: whether to append observed burnin states to video reconstruction
    Returns:
      ep_rewards - [batch_size x ep_length], intrinsic reward for each step in an episode
    """
    name = f'ep_{start_imagining_at}_{burnin}_{imagine_for}_{include_truth_images}_{show_burnin_recon}'
    fn_gif = f'{save_dir}/{name}.gif'
    fn_img = f'{save_dir}_{expid}_{name}_imgloss.{fig_suffix}'
    print(fn_gif)

    b = 1 # batchsize
    wm = agnt.wm
    data = wm.preprocess(data)
    embed = wm.encoder(data)
    start_observing_at = start_imagining_at-burnin
    end_imagining_at = start_imagining_at+imagine_for
    if imagine_for == -1:
      end_imagining_at = -1

    # Get posterior (using the visual data)
    post, _ = wm.rssm.observe(embed[:b,
                                  start_observing_at:start_imagining_at],
                                data['action'][:b,
                                  start_observing_at:start_imagining_at],
                                sample=(not deterministic_wm))

    init = {k: v[:, -1] for k, v in post.items()}
    prior = wm.rssm.imagine(data['action'][:b,
                            start_imagining_at:end_imagining_at],
                            init,
                            sample=(not deterministic_wm))
    openl = wm.heads['image'](wm.rssm.get_feat(prior)).mode()

    burnin_recon = wm.heads['image'](
      wm.rssm.get_feat(post)).mode()[:b]
    model_w_burnin = tf.concat([burnin_recon[:, :burnin] + 0.5, openl + 0.5], 1)
    if show_burnin_recon:
      model = model_w_burnin
    else:
      model = openl + 0.5

    if include_truth_images:
      truth_w_burnin = data['image'][:b,
              start_observing_at:end_imagining_at] + 0.5
      if show_burnin_recon:
        truth = truth_w_burnin
      else:
        truth = data['image'][:b,
                start_imagining_at:end_imagining_at] + 0.5

      error = (model - truth + 1) / 2
      video = tf.concat([truth, model, error], 2)
      if do_include_error_img:
        error_w_burnin = (model_w_burnin - truth_w_burnin + 1) / 2
        # error_w_burnin = 1 - np.abs((truth_w_burnin - model_w_burnin) / 2)

        abs_error_w_burnin = np.abs((truth_w_burnin - model_w_burnin))
        error_w_burnin = 1 - abs_error_w_burnin
        img_abserr_w_burnin = abs_error_w_burnin.mean(-1).mean(-1).mean(-1)
        # plt.imshow(error_w_burnin[0, 0, :].astype(np.float32))

        squared_error_w_burnin = (model_w_burnin.numpy() - truth_w_burnin.numpy()) ** 2
        error_img_w_burnin = 1 - squared_error_w_burnin
        img_mse_w_burnin = squared_error_w_burnin.mean(-1).mean(-1).mean(-1)
        # plt.imshow(squared_error_w_burnin[0, 0, :].astype(np.float32))

        video_w_burnin = tf.concat([truth_w_burnin, model_w_burnin, error_w_burnin], 2)
      else:
        video_w_burnin = tf.concat([truth_w_burnin, model_w_burnin], 2)
        img_mse_w_burnin = None
        img_abserr_w_burnin = None
    else:
      video = model
      video_w_burnin = model_w_burnin
      img_mse_w_burnin = None
      img_abserr_w_burnin = None

    B, T, H, W, C = video.shape
    imgs = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
    imgs = imgs.numpy()
    imgs = np.clip(imgs, 0, 1)
    imgs *= 255
    imgs = imgs.astype(np.int)

    if do_save_gif:
      from moviepy.editor import ImageSequenceClip
      clip = ImageSequenceClip(list(imgs), fps=20)
      clip.write_gif(fn_gif, fps=20)

    if plot_intrinsic_reward:
      # Get curiosity for the burnin (post) and then imagined (prior)
      all_intr_reward = []
      for start in (post, prior):
        # For each start state, do some imagined exploration
        (feat, state, ## state is: for t in horizon; for each starting latent state (from each timepoint of each batch); the latent state
         action, disc) = agnt.wm.imagine(agnt._expl_behavior.ac.actor,
                                         start,
                                         agnt.config.imag_horizon,
                                         deterministic=deterministic_expl)
        rewards = agnt._expl_behavior._intr_reward(feat, state, action)
        intr_reward = rewards[1]
        intr_reward = intr_reward.numpy()
        all_intr_reward.append(intr_reward)
      intr_reward = np.hstack(all_intr_reward)
      discounted_rewards = get_discounted_rewards(intr_reward, discounts)

    # Compute image loss
    if plot_image_loss:
      B, T, H, W, C = video_w_burnin.shape
      imgs_w_burnin = video_w_burnin.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
      imgs_w_burnin = imgs_w_burnin.numpy()
      imgs_w_burnin = np.clip(imgs_w_burnin, 0, 1)
      imgs_w_burnin *= 255
      imgs_w_burnin = imgs_w_burnin.astype(np.int)

      # Compute losses
      truth_imagine = data['image'][:b,
                      start_imagining_at:end_imagining_at]
      like_imagine = tf.cast(wm.heads['image'](wm.rssm.get_feat(prior)).log_prob(truth_imagine), tf.float32)
      loss_imagine = -like_imagine.numpy()

      truth_burnin = data['image'][:b,
                     start_observing_at:start_imagining_at]
      like_burnin = tf.cast(wm.heads['image'](wm.rssm.get_feat(post)).log_prob(truth_burnin), tf.float32)
      loss_burnin = -like_burnin.numpy()

      if not no_plots:
        # Make summary plot
        if key_t is None:
          t = np.array([-10, 0, 5, 10, 15, 20, imagine_for - 1])
        else:
          t = key_t

        all_loss = np.hstack((loss_burnin[0], loss_imagine[0]))
        if plot_intrinsic_reward:
          nrows = 4
        else:
          nrows = 3
        plt.figure(figsize=(10, nrows*2))
        if plot_intrinsic_reward:
          plt.subplot2grid((nrows, len(t)), (0, 0), colspan=len(t))
          p2, = plt.plot(np.arange(-burnin, imagine_for), np.squeeze(discounted_rewards[0]), 'b')
          plt.ylabel('Intr reward')
          plt.xlabel('Imagined steps')
          [plt.axvline(x, linestyle='--', color='k') for x in t]
          if ylim_intr_rew is not None:
            plt.ylim(ylim_intr_rew)
        plt.subplot2grid((nrows, len(t)), (int(plot_intrinsic_reward), 0), colspan=len(t))
        p1, = plt.plot(np.squeeze(loss_imagine))
        p2, = plt.plot(np.arange(-burnin, 0), np.squeeze(loss_burnin), 'r')
        plt.legend([p2], ['observed'])
        plt.ylabel('Image loss')
        plt.xlabel('Imagined steps')
        [plt.axvline(x, linestyle='--', color='k') for x in t]
        if ylim_imgloss is not None:
          plt.ylim(ylim_imgloss)
        for i in range(len(t)):
          plt.subplot2grid((nrows, len(t)), (1 + int(plot_intrinsic_reward), i), rowspan=2)
          plt.imshow(imgs_w_burnin[t[i]+burnin])
          plt.title(f't={t[i]}: {all_loss[t[i]+burnin]:0.0f}')
          plt.axis('off')
        plt.suptitle(':'.join([save_dir.split('/')[-5], save_dir.split('/')[-1], save_dir.split('/')[-2]]))
        plt.tight_layout()
        plt.savefig(fn_img)
    else:
      loss_imagine = None
      loss_burnin = None

    if plot_intrinsic_reward:
      intr_rew = np.squeeze(discounted_rewards[0])
    else:
      intr_rew = None
    if do_return_imgs:
      return loss_imagine, loss_burnin, intr_rew, imgs_w_burnin, img_mse_w_burnin, img_abserr_w_burnin
    else:
      return loss_imagine, loss_burnin, intr_rew
    print('done')

def convert_matplotlib_fig_to_array(fig):
  fig.canvas.draw()

  # Now we can save it to a numpy array.
  data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  return data


def amount_of_color(images, color_min, color_max):
  scale_factor = 1 # If 1, then result is fraction of pixels that are the target color.
  if len(images.shape) == 4:
    images = images[np.newaxis, :]
  npixels = images.shape[-2]*images.shape[-3]
  r = tf.zeros((images.shape[0], images.shape[1]), dtype=tf.float32)
  color = tf.where((images[:, :, :, :, 0] >= color_min[0]) &
                   (images[:, :, :, :, 1] >= color_min[1]) &
                   (images[:, :, :, :, 2] >= color_min[2]) &
                   (images[:, :, :, :, 0] <= color_max[0]) &
                   (images[:, :, :, :, 1] <= color_max[1]) &
                   (images[:, :, :, :, 2] <= color_max[2])
                   )

  start = 0
  for b in range(images.shape[0]):
    batch_inds = tf.where(color[:, 0] == b)
    if tf.size(batch_inds) > 0:
      # tf.print(f'INSIDE THE LOOP', output_stream=sys.stderr),
      stop = start + len(batch_inds)
      batch_color = tf.cast(color[start:stop, 1], dtype=tf.int32)
      # tf.print(f'max batch_color: ', output_stream=sys.stderr), tf.print(tf.math.reduce_max(batch_color), output_stream=sys.stderr)
      # tf.print('min batch_color: ', output_stream=sys.stderr), tf.print(tf.math.reduce_min(batch_color), output_stream=sys.stderr)
      batch_r = tf.math.bincount(batch_color)
      # tf.print(f'batch_r: ', output_stream=sys.stderr), tf.print(batch_r, output_stream=sys.stderr)
      batch_r = tf.cast(batch_r, dtype=tf.float32) / (npixels / scale_factor) # Scaling since target ball can't take much more than half the screen
      # tf.print(f'batch_r float32: ', output_stream=sys.stderr), tf.print(batch_r, output_stream=sys.stderr)
      # tf.print(f'tf.shape(batch_r): ', output_stream=sys.stderr), tf.print(tf.shape(batch_r), output_stream=sys.stderr)
      batch_shape = tf.shape(batch_r)
      # tf.print(f'batch_shape: ', output_stream=sys.stderr), tf.print(batch_shape, output_stream=sys.stderr)
      # tf.print(f'batch_shape[0]: ', output_stream=sys.stderr), tf.print(batch_shape[0], output_stream=sys.stderr)
      # tf.print(f'batch_shape[0].dtype: ', output_stream=sys.stderr), tf.print(batch_shape[0].dtype, output_stream=sys.stderr)
      # tf.print(f'tf.shape(r): ', output_stream=sys.stderr), tf.print(tf.shape(r), output_stream=sys.stderr)

      indices = tf.stack([b*tf.ones(batch_shape[0], dtype=tf.int32), tf.range(batch_shape[0])], axis=1)
      r = tf.tensor_scatter_nd_add(r, indices, batch_r)

      start = stop
  r = r.astype(tf.float32)

  # import matplotlib.pyplot as plt
  # plt.imshow(r, aspect='auto'), plt.colorbar(), plt.show()
  # tf.print(f'max _amount_of_color: ', output_stream=sys.stderr), tf.print(tf.math.reduce_max(r), output_stream=sys.stderr)
  # tf.print('min _amount_of_color: ', output_stream=sys.stderr), tf.print(tf.math.reduce_min(r), output_stream=sys.stderr)

  return r

