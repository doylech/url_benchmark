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
  parser = argparse.ArgumentParser(description='Specify episode arguments.')
  parser.add_argument('--bufferid', default='EXAMPLE_EPS_black')
  parser.add_argument('--expid',    default='DRA-159')


  args = parser.parse_args()
  expid = args.expid # For Fig 1: 'DRE-354'   # 'DRE-377'
  bufferid = args.bufferid


  # checkpoint_name = 'variables_train_agent_envindex0_final_000502500.pkl'
  checkpoint_name = 'variables_pretrained_env0.pkl'
  checkpoint_name = 'variables_train_agent_envindex0_000007500.pkl'
  checkpoint_name = 'variables_train_agent_envindex0_000017500.pkl'
  checkpoint_name = 'variables_train_agent_envindex0_000500000.pkl'
  checkpoint_name = 'variables_train_agent_envindex0_001452500.pkl'
  checkpoint_name = 'variables_train_agent_envindex0_001502500.pkl' # For Fig 1.
  checkpoint_name = 'variables_train_agent_envindex0_000015000.pkl' #
  checkpoint_name = 'variables_train_agent_envindex0_000015500.pkl' #
  # checkpoint_name = 'variables_train_agent_envindex0_000030000.pkl' #
  # checkpoint_name = 'variables_train_agent_envindex0_000030500.pkl' #
  # checkpoint_name = 'variables_train_agent_envindex0_000041500.pkl'

  # which_eps = [[98], [99], [100],[101]]
  which_eps = [[0], [1], [2], [3], [4]]
  start_imagining_at = 220 #263   ### 30 FOR DEBUGGING
  if 'GEN-EXAMPLE_EPS' in bufferid:
    which_eps = [[0], [1], [2], [3]]
    start_imagining_at = 36

  plot_intrinsic_reward = False
  replay_buffer_name = 'train_replay_0'

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
    all_loss = []
    all_imgs_w_burnin = []
    all_intr_reward = []
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

      loss_imagine, loss_burnin, intr_reward, imgs_w_burnin, img_mse_w_burnin, img_abserr_w_burnin  = \
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
                                           )
      all_loss.append(np.hstack((loss_burnin, loss_imagine)))
      all_intr_reward.append(intr_reward)
      all_imgs_w_burnin.append([imgs_w_burnin[t+burnin] for t in key_t])


  do_plot_img_comparison = False
  if do_plot_img_comparison:
    for i in range(len(all_imgs_w_burnin)):
      imgs_w_burnin = all_imgs_w_burnin[i]
      nimgs = len(np.where(key_t>0)[0])
      plt.figure()
      for ii in range(nimgs):
        plt.subplot(3, nimgs, ii+1)
        plt.imshow
        plt.subplot(3, nimgs, ii+nimgs)

        plt.subplot(3, nimgs, ii+2*nimgs)




  save_dir = f"{plot_dir}/{bufferid}/{checkpoint_name}"
  plt.figure()
  for i, ep  in enumerate(which_eps):
    plt.plot(np.arange(-burnin, imagine_for), all_loss[i][0])
  plt.legend(which_eps)
  plt.ylabel('Image Loss')
  plt.xlabel('Imagined step')
  plt.suptitle(':'.join([expid, bufferid, checkpoint_name]), fontsize=10)
  plt.savefig(os.path.join(save_dir, 'img_loss.png'))
  print(f"saved to {os.path.join(save_dir, 'img_loss.png')}")
  plt.show()

  to_save = {}
  to_save['all_loss'] = all_loss
  to_save['bufferid'] = bufferid
  to_save['which_eps'] = which_eps
  to_save['start_imagining_at'] = start_imagining_at
  to_save['replay_buffer_name'] = replay_buffer_name
  to_save['burnin'] = burnin
  to_save['imagine_for'] = imagine_for
  to_save['imgs_t'] = key_t
  to_save['all_imgs_w_burnin'] = all_imgs_w_burnin
  to_save['all_intr_reward'] = all_intr_reward
  pickle.dump(to_save, open(os.path.join(save_dir, 'img_loss.pkl'), 'wb'))
  print('done')