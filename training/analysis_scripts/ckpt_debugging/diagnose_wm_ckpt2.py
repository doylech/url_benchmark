"""Load in a WM checkpoint to see why it is crashing."""

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
  # expid = 'p2e_fiddle_dreamer-launch'
  expid = 'GEN-801'
  bufferid = 'GEN-EXAMPLE_EPS'
  # checkpoint_name = 'post_init.pkl'
  checkpoint_name = 'variables_pretrained_env0.pkl'
  # checkpoint_name = 'variables_train_agent_envindex0_000032500.pkl'
  wm_checkpoint = 'crash_wm.pkl'
  replay_buffer_name = 'eval_replay_0'

  home = expanduser("~")
  basedir = f"{home}/logs/{expid}"
  buffer_basedir = f"{home}/logs/{bufferid}"
  plot_lims=[-20,20]

  plot_dir = f'{basedir}/plots'
  os.makedirs(plot_dir, exist_ok=True)

  # from tensorflow.keras.mixed_precision import experimental as prec
  # prec.set_policy(prec.Policy('mixed_float16'))

  ## Load agent from checkpoint
  agnt = lu.load_agent(basedir,
                       checkpoint_name=checkpoint_name,
                       batch_size=2,
                       deterministic=False
                       )
  print(agnt.wm.trainable_variables[1][0, 0,0, :10])
  agnt.wm.load(os.path.join(basedir, wm_checkpoint))
  print(agnt.wm.trainable_variables[1][0, 0,0, :10])

  do_save_wm = 1
  if do_save_wm:
    wm_vars = dict()
    for i in range(len(agnt.wm.trainable_variables)):
      wm_vars[i] = agnt.wm.trainable_variables[i].numpy()
    with open(os.path.join(basedir, 'wm_vars.pkl'), 'wb') as f:
      pickle.dump(wm_vars, f, protocol=pickle.HIGHEST_PROTOCOL)
    agent_vars = dict()
    for i in range(len(agnt.trainable_variables)):
      agent_vars[i] = agnt.trainable_variables[i].numpy()
    with open(os.path.join(basedir, 'agent_vars.pkl'), 'wb') as f:
      pickle.dump(agent_vars, f, protocol=pickle.HIGHEST_PROTOCOL)

  # Now run the databatch through, does it crash?
  with open(os.path.join(basedir, 'crash_data.pkl'), 'rb') as f:
    data = pickle.load(f)

  with open(os.path.join(basedir, 'crash_state.pkl'), 'rb') as f:
    sf = pickle.load(f)
    state_o = sf['state']
    feat_o = sf['feat']
    post_o = sf['post']
    like_o = sf['like']
    name_o = sf['name']
    embed_o = sf['embed']
  # plt.imshow(data['image'][2, 0, :].astype(np.float32)+0.5)
  # plt.show()

  # state = None
  state = state_o
  self = agnt.wm
  # data = self.preprocess(data) # Data has already been preprocessed

  ##### debugging ####
  data_const = {'image': tf.ones_like(data['image'])}
  embed_const = self.encoder(data_const)
  print('embed_const'), print(embed_const[0,0,:50])
  ##### debugging ####

  embed = self.encoder(data)
  post, prior = self.rssm.observe(embed, data['action'], state, sample=False)

  print('embed'), print(embed[0,9,0])
  print('embed_o'), print(embed_o[0,9,0]) # TODO!!
  embed_small = self.encoder({'image': data['image'][0, 9, :]})

  print('data_action'),  print(data['action'][0, :, 0])
  print('data_image'),   print(data['image'][0, :, 0, 0, 0])
  print('embed_o'), print(embed_o[0,:,0])
  print('embed'), print(embed[0,:,0])
  print('embed-embed_o'), print(embed[0,:,0] - embed_o[0,:,0])

  print(f'Embed - embed_o: \t {np.max((embed- embed_o).numpy())}')
  post_1, prior_1 = self.rssm.observe(embed_o, data['action'], state, sample=False)
  print(f"post  - post_o: \t {np.max((post['deter'] - post_o['deter']).numpy())}")
  # print(np.max((post['deter'] - post_1['deter']).numpy()))

  feat = self.rssm.get_feat(post)
  print(f'feat  - feat_o: \t {np.max((feat- feat_o).numpy())}')

  # post = post_o

  raise('done')
  name = 'image'
  head = self.heads[name]
  inp = feat_o
  i = head(inp).mode().astype(np.float32)
  like = tf.cast(head(inp).log_prob(data[name]), tf.float32)

  i2 = agnt2.wm.heads['image'](feat_o).astype(np.float32)

  bi = [5, 0] # bad ind
  # bi = [3, 45] # bad ind
  plt.imshow(i[4, 0, :] + 0.5), plt.show()
  plt.imshow(i[4, 1, :] + 0.5), plt.show()
  plt.imshow(i[3, 0, :] + 0.5), plt.show()
  plt.imshow(i[bi[0], bi[1], :] + 0.5), plt.show()

  # Try swapping parts of the feat vector
  inp_new = inp._copy().numpy()
  # inp_new[3, 45, -200:] = inp_new[3, 5, -200:]
  # inp_new[3, 45, :-200] = inp_new[3, 5, :-200]
  i_new = head(inp_new).mode().astype(np.float32)
  plt.imshow(i_new[bi[0], bi[1], :] + 0.5), plt.show()

  # Is there anything weird with the image 'head'
  all_vars = [tf.reshape(x, [-1]).numpy() for x in head.trainable_variables]
  all_vars = np.hstack(all_vars)
  plt.plot(all_vars.astype(np.float32)), plt.show()

  # Is there anything weird with the feat
  plt.plot(tf.reshape(feat_o, [-1]).numpy().astype(np.float32)), plt.show()
  plt.imshow(feat_o.numpy()[bi[0], :].astype(np.float32), aspect='auto', interpolation='nearest'), \
  plt.xlabel('Feature dimension'), plt.ylabel('Timepoint in episode')
  plt.axhline(bi[1]), plt.show()

  not_inf = np.where(~np.isinf(like))
  inf = np.where(np.isinf(like))
  plt.plot(feat_o.numpy()[inf[0], inf[1]].max(axis=0).astype(np.float32) -
           feat_o.numpy()[not_inf[0], not_inf[1]].max(axis=0).max(axis=0).astype(np.float32))
  plt.show()
  plt.plot(feat_o.numpy()[bi[0], 45:].max(axis=0).astype(np.float32) -
           feat_o.numpy()[:bi[0], :40].max(axis=0).max(axis=0).astype(np.float32))



  kl_loss, kl_value = self.rssm.kl_loss(post, prior, **self.config.kl)
  assert len(kl_loss.shape) == 0
  likes = {}
  losses = {'kl': kl_loss}
  feat = self.rssm.get_feat(post)
  for name, head in self.heads.items():
    grad_head = (name in self.config.grad_heads)
    inp = feat if grad_head else tf.stop_gradient(feat)
    like = tf.cast(head(inp).log_prob(data[name]), tf.float32)
    likes[name] = like
    losses[name] = -like.mean()
    print(losses[name])

  print('done')