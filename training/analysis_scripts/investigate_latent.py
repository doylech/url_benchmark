"""
Load a various model checkpoints, investigate how the latent state representation
changes during training (on both familiar and new ball)
Potentially use hand-crafted test episodes from envs.scripts.make_example_episodes.py
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
from collections import defaultdict

tf.config.run_functions_eagerly(True)


if __name__ == "__main__":
  ## Set options for script.
  expid = 'GEN-616'
  bufferid = 'GEN-EXAMPLE_EPS'
  checkpoint_names = ['variables_pretrained_env0.pkl',
                      'variables_train_agent_envindex0_000003500.pkl',
                      'variables_train_agent_envindex0_000004500.pkl',
                      'variables_train_agent_envindex0_000005500.pkl',
                      'variables_train_agent_envindex0_000006500.pkl',
                      'variables_train_agent_envindex0_000007500.pkl',
                      'variables_train_agent_envindex0_000008500.pkl',
                      'variables_train_agent_envindex0_000009500.pkl',
                      'variables_train_agent_envindex0_000010500.pkl',
                      ]
  replay_buffer_name = 'train_replay_0'
  just_load = False

  start_observing_at = 0
  end_observing_at = 100
  deterministic_wm = True


  home = expanduser("~")
  basedir = f"{home}/logs/{expid}"
  buffer_basedir = f"{home}/logs/{bufferid}"
  plot_lims=[-20,20]

  plot_dir = f'{basedir}/plots'
  os.makedirs(plot_dir, exist_ok=True)
  savedir = f'{basedir}/investigate_latent'
  os.makedirs(savedir, exist_ok=True)

  from tensorflow.keras.mixed_precision import experimental as prec
  prec.set_policy(prec.Policy('mixed_float16'))

  if not just_load:
    posts = defaultdict(list)
    priors = defaultdict(list)
    recons = defaultdict(list)

    for checkpoint_name in checkpoint_names:
      ## Load agent from checkpoint
      print(checkpoint_name)
      agnt = lu.load_agent(basedir,
                           checkpoint_name=checkpoint_name,
                           batch_size=5,
                           deterministic=False
                           )

      do_visualize_dream = True


      which_eps = [[0],[1],[2],[3]] # Select the episodes from the replay buffer
      if do_visualize_dream:
        all_loss = []
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

          b = 1  # batchsize
          wm = agnt.wm
          data = wm.preprocess(data)
          embed = wm.encoder(data)

          # Get posterior (using the visual data)
          post, prior = wm.rssm.observe(embed[:b,
                                        start_observing_at:end_observing_at],
                                        data['action'][:b,
                                        start_observing_at:end_observing_at],
                                        sample=(not deterministic_wm))

          recon = wm.heads['image'](
            wm.rssm.get_feat(post)).mode()[:b].numpy().astype(np.float32) + 0.5


          posts[batch_eps[0]].append(post)
          priors[batch_eps[0]].append(prior)
          recons[batch_eps[0]].append(recon)
          print('done')

    pickle.dump({'posts': posts, 'priors': priors, 'recons': recons,
                 'expid': expid,
                 'bufferid': bufferid, 'checkpoint_names': checkpoint_names,
                 'replay_buffer_name': replay_buffer_name,
                 'start_observing_at': start_observing_at,
                 'end_observing_at': end_observing_at,
                 'deterministic_wm': deterministic_wm},
                open( f"{savedir}/latents.pkl", "wb" ) )
    print('done')

  dd = pickle.load(open(f"{savedir}/latents.pkl", "rb"))
  posts = dd['posts']
  priors = dd['priors']

  plotdir = f"{savedir}/plots"
  os.makedirs(plotdir, exist_ok=True)
  do_save_plot = True

  for tt in [30, 40, 50]:
    for which_ep in [0, 1, 2, 3]:
      ep_posts = posts[which_ep]
      ep_deter = np.vstack([x['deter'].numpy().astype(np.float32) for x in ep_posts])
      ep_recons = np.vstack(recons[which_ep])

      do_vector = False
      if do_vector:
        plt.figure()
        plt.imshow(np.squeeze(ep_deter[:, tt, :]), aspect='auto', interpolation='nearest')
        plt.yticks(np.arange(len(checkpoint_names)), checkpoint_names)
        plt.title(f'Ep {which_ep}, t {tt}')
        if do_save_plot:
          plt.savefig(f"{plotdir}/vector_{tt}_{which_ep}.png")
        plt.show()

      plt.figure()
      plt.imshow(np.corrcoef(np.squeeze(ep_deter[:, tt, :])), aspect='auto', interpolation='nearest')
      plt.clim(0.9, 1)
      plt.colorbar()
      plt.yticks(np.arange(len(checkpoint_names)), checkpoint_names)
      plt.title(f'Ep {which_ep}, t {tt}')
      if do_save_plot:
        plt.savefig(f"{plotdir}/corr_{tt}_{which_ep}.png")
      plt.show()

      plt.figure()
      for i in range(ep_recons.shape[0]):
        nn = int(np.ceil(np.sqrt(ep_recons.shape[0])))
        plt.subplot(nn, nn, i+1)
        plt.imshow(ep_recons[i, tt, :])
        plt.axis('off')
        plt.title(i)
      plt.suptitle(f'Ep {which_ep}, t {tt}')
      if do_save_plot:
        plt.savefig(f"{plotdir}/recon_{tt}_{which_ep}.png")
      plt.show()
  print('done')

