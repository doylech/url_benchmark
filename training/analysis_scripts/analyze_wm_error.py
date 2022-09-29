"""
Load a model, investigate what it is dreaming.
Potentially use hand-crafted test episodes from envs.scripts.make_example_episodes.py

See download_ckpt.sh to get necessary files from remote server.
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
import scipy.stats

import elements
import common
from dreamerv2 import agent
import argparse
from tqdm import tqdm

from os.path import expanduser
import training.utils.loading_utils as lu
import training.utils.diagnosis_utils as du

tf.config.run_functions_eagerly(True)

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42

def simple_plot(ax):
  ax.spines.right.set_visible(False)
  ax.spines.top.set_visible(False)
  ax.yaxis.set_ticks_position('left')
  ax.xaxis.set_ticks_position('bottom')

if __name__ == "__main__":
  ids = {  # freerange, black
    'DRE-354': 't4-tf-2', # black
    'DRE-346': 't4-tf-8', # black
    'DRE-348': 't4-tf-9', # black
    'DRE-475': 't4-tf-9',  # black
  }
  # ids = {  # random policy playground
  #   'DRE-376': 't4-tf-2',
  #   'DRE-375': 't4-tf-1',
  #   'DRE-378': 't4-tf-2',
  #   'DRE-377': 't4-tf-1',
  # }

  # First four are p2e, last four are random
  expids = ['DRE-354', 'DRE-346', 'DRE-348', 'DRE-475', 'DRE-376', 'DRE-375', 'DRE-378', 'DRE-377', ] # p2e

  do_baseline_subtraction = False  # Set error=0 during burnin (non-imagination)

  bufferid = 'GEN-EXAMPLE_EPS_black'
  which_eps = [[1], [3]]
  replay_buffer_name = 'train_replay_0'

  # checkpoints = np.arange(1, 16)*1e5 + 2500
  checkpoints = np.arange(1, 16, 1)*1e5 + 2500

  checkpoints = checkpoints.astype(int)

  t0 = time.time()

  all_expt_loss = {}
  all_expt_intr_reward = {}
  all_expt_ckpt = {}
  all_expt_which_eps = {}

  do_just_load = True

  if not do_just_load:
    for expid in expids:
      all_loss = []
      all_mse = []
      all_abserr = []
      all_imgs_w_burnin = []
      all_intr_reward = []
      all_ckpt = []
      all_which_eps = []
      for checkpoint in tqdm(checkpoints):
        checkpoint_name = f'variables_train_agent_envindex0_{checkpoint:09}.pkl'
        print(checkpoint_name)

        start_imagining_at = 36

        plot_intrinsic_reward = False
        burnin = 30
        imagine_for = 45
        ylim_intr_rew = None

        ylim_intr_rew=[-3.5, -1.5]
        # ylim_intr_rew=[0, 0.0005]
        # ylim_intr_rew = [0, 0.05]
        # ylim_intr_rew = [0, 0.001]


        home = expanduser("~")
        basedir = f"{home}/logs/{expid}"
        buffer_basedir = f"{home}/logs/{bufferid}"
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

        do_visualize_dream = True


        if do_visualize_dream:
          for batch_eps in which_eps:

            eps = lu.load_eps(buffer_basedir, replay_buffer_name, batch_eps=batch_eps)

            nt = agnt.config.dataset.length
            nt = eps[0]['action'].shape[0]

            xyz = np.stack([ep['absolute_position_agent0'][:nt, :] for ep in eps], axis=0)
            data = {key: np.vstack([np.expand_dims(ep[key].astype('float32'), axis=0)
                                    for ep in eps])[:, :nt]
                    for key in eps[0].keys()}
            data = {key: tf.cast(data[key], dtype=data[key].dtype) for key in data.keys()}

            # Optionally, overwrite actions here, by overwriting data['action']

            save_dir = f"{plot_dir}/{bufferid}/{checkpoint_name}/{batch_eps[0]}"
            os.makedirs(save_dir, exist_ok=True)

            actions_for_imagine = data['action']
            # key_t = np.array([-10, -3, 0, 5, 10, 15, 20])
            # key_t = np.array([-10, -3, 0, 5, 10, 15, 20, 25, 30])
            key_t = np.array([0, 3, 6, 9, 12, 15, 18])

            loss_imagine, loss_burnin, intr_reward, imgs_w_burnin, img_mse_w_burnin, img_abserr_w_burnin = \
                                                 du.dream(agnt, data,
                                                 actions_for_imagine, save_dir,
                                                 expid=expid,
                                                 start_imagining_at=start_imagining_at,
                                                 burnin=burnin,
                                                 imagine_for=imagine_for,
                                                 plot_intrinsic_reward=plot_intrinsic_reward,
                                                 show_burnin_recon=False,
                                                 deterministic_wm=True,
                                                 do_save_gif=False,
                                                 ylim_imgloss=[11200, 12300],
                                                 ylim_intr_rew=ylim_intr_rew,
                                                 key_t=key_t,
                                                 do_return_imgs=True,
                                                 do_include_error_img=True,
                                                 fig_suffix='pdf',
                                                 no_plots=True,
                                                 )
            all_mse.append(img_mse_w_burnin)
            all_abserr.append(img_abserr_w_burnin)
            all_loss.append(np.hstack((loss_burnin, loss_imagine)))
            all_intr_reward.append(intr_reward)
            all_ckpt.append(checkpoint_name)
            all_which_eps.append(which_eps)
            # all_imgs_w_burnin.append([imgs_w_burnin[t+burnin] for t in key_t])

      save_dir = f"{plot_dir}/{bufferid}/"
      plt.figure()
      for i, ep in enumerate(which_eps):
        plt.plot(np.arange(-burnin, imagine_for), all_loss[i][0])
      plt.legend(which_eps)
      plt.ylabel('Image Loss')
      plt.xlabel('Imagined step')
      plt.suptitle(':'.join([expid, bufferid, checkpoint_name]), fontsize=10)
      plt.savefig(os.path.join(save_dir, 'img_loss.png'))
      plt.show()

      to_save = {}
      to_save['checkpoints'] = all_ckpt
      to_save['all_loss'] = all_loss
      to_save['all_mse'] = all_mse
      to_save['all_abserr'] = all_abserr
      to_save['bufferid'] = bufferid
      to_save['which_eps'] = which_eps
      to_save['all_which_eps'] = all_which_eps
      to_save['start_imagining_at'] = start_imagining_at
      to_save['replay_buffer_name'] = replay_buffer_name
      to_save['burnin'] = burnin
      to_save['imagine_for'] = imagine_for
      to_save['imgs_t'] = key_t
      to_save['all_imgs_w_burnin'] = all_imgs_w_burnin
      to_save['all_intr_reward'] = all_intr_reward
      pickle.dump(to_save, open(os.path.join(save_dir, 'img_loss_all_ckpt.pkl'), 'wb'))
      print(f'Saved to {save_dir}')

      all_expt_loss[expid] = all_loss
      all_expt_intr_reward[expid] = all_intr_reward
      all_expt_ckpt[expid] = all_ckpt
      all_expt_which_eps[expid] = all_which_eps

  print(f'{time.time() - t0}')
  print('done')

  # Load and make plots
  all_mean_losses = []
  all_baseline_losses = []
  for expid in expids:
    plt.figure()
    home = expanduser("~")
    basedir = f"{home}/logs/{expid}"
    buffer_basedir = f"{home}/logs/{bufferid}"
    plot_dir = f'{basedir}/plots'
    load_dir = f"{plot_dir}/{bufferid}/"
    d = pickle.load(open(os.path.join(load_dir, 'img_loss_all_ckpt.pkl'), 'rb'))

    which_loss = 'mse' # 'mse', 'abs', 'loss'
    if which_loss == 'loss':
      all_expt_loss[expid] = d['all_loss']
      y_str = 'Image loss'
    elif which_loss == 'mse':
      all_expt_loss[expid] = d['all_mse']
      y_str = 'MSE'
    elif which_loss == 'abs':
      all_expt_loss[expid] = d['all_abserr']
      y_str = 'Abs(error)'


    all_expt_intr_reward[expid] = d['all_intr_reward']
    all_expt_ckpt[expid] = d['checkpoints']
    all_expt_which_eps[expid] = d['all_which_eps']
    burnin = d['burnin']
    imagine_for = d['imagine_for']

    all_loss = all_expt_loss[expid]
    all_ckpt = all_expt_ckpt[expid]
    plt.figure()
    legend_str = []

    cmap = matplotlib.cm.get_cmap('plasma')

    mean_losses = []
    baseline_losses = []
    for i in np.arange(1, len(all_loss), 2):
      mean_losses.append(np.mean(all_loss[i][0][burnin:burnin+15]))
      baseline_losses.append(np.mean(all_loss[i][0][burnin-15:burnin]))
      # for i, ep in enumerate(which_eps):
      plt.plot(np.arange(-burnin, imagine_for), all_loss[i][0], color=cmap(i/len(all_loss)))
      plt.ylabel(f'{y_str}')
      plt.xlabel('Imagined step')
      legend_str.append(all_ckpt[i][-13:-4])
      plt.suptitle(':'.join([expid, bufferid, all_ckpt[i]]), fontsize=10)
      # plt.savefig(os.path.join(save_dir, 'img_loss.png'))
    # plt.legend(legend_str)
    plt.xlim([0, 15])
    plt.show()

    all_mean_losses.append(mean_losses)
    all_baseline_losses.append(baseline_losses)
    # TODO: Sum error across [0,15] and then plot vs checkpoint (and aggregate)

  all_mean_losses = np.array(all_mean_losses)
  all_baseline_losses = np.array(all_baseline_losses)
  plt.plot(all_mean_losses.T)
  plt.show()


  plt.figure()
  tt = checkpoints*2

  if do_baseline_subtraction:
    p2e_vals = all_mean_losses[:4, :] - all_baseline_losses[:4, :]
  else:
    p2e_vals = all_mean_losses[:4, :]
  mm = np.mean(p2e_vals, axis=0)
  ss = scipy.stats.sem(p2e_vals, axis=0)
  plt.fill_between(tt, mm-ss, mm+ss, alpha=0.5)
  p1, = plt.plot(tt, mm)

  if do_baseline_subtraction:
    rand_vals = all_mean_losses[4:, :] - all_baseline_losses[4:, :]
  else:
    rand_vals = all_mean_losses[4:, :]
  mm = np.mean(rand_vals, axis=0)
  ss = scipy.stats.sem(rand_vals, axis=0)
  plt.fill_between(tt, mm-ss, mm+ss, alpha=0.5)
  p2, = plt.plot(tt, mm)
  plt.xlabel('Environment step #')
  plt.ylabel(f'{y_str}')
  plt.title('Error on test episode')
  plt.legend([p1, p2], ['Intr. motiv.', 'Random'], frameon=False)
  simple_plot(plt.gca())

  plot_dir = f"/home/saal2/logs/plots"
  plt.savefig(os.path.join(plot_dir, f'test_ep_error_{expids}.pdf'))
  plt.show()

  print('done')