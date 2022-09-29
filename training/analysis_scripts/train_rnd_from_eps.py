"""
Train an RND agent from episodes in a replay buffer.
Test on a specified set of images (see generate_test_images_for_novelty.py).
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
import glob

from tqdm import tqdm

from os.path import expanduser
import training.utils.loading_utils as lu
import training.utils.diagnosis_utils as du
import training.utils.img_utils as iu
import common

tf.config.run_functions_eagerly(True)

class RND_raw(common.Module):
  def __init__(self, config):
    self.config = config
    self.target_network = common.ConvEncoderMLP(**self.config['rnd_recon_target_head'])
    self.predictor_network = common.ConvEncoderMLP(**self.config['rnd_recon_predictor_head'])
    self.opt = common.Optimizer('RNDpred', **self.config['expl_opt'])
    self.ep_iter = 0

  def _train_predictor_ep(self, ep, num_iter=1, logger=None, step=None, train_every=1):
    if train_every > 0 and np.mod(self.ep_iter, train_every) == train_every - 1:
      print(f'Training rnd_predictor, ep: {self.ep_iter}')
      eps = [ep]
      data = {key: np.vstack([np.expand_dims(ep[key].astype('float32'), axis=0) for ep in eps])
              for key in ['image', 'reward', 'discount', 'action']}
      # model_loss, state, outputs, mets = self.wm.loss(data, state=None, sample=False)  # Should we sample???
      metrics = self._train_predictor(data['image'], num_iter=num_iter)
    self.ep_iter += 1
    return metrics

  def _train_predictor(self, img, num_iter=1):
    metrics = {}
    s_t = {'image': img}
    for i in range(num_iter):
      target = self.target_network(s_t)
      with tf.GradientTape() as tape:
        mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
        loss = tf.cast(mse(target, self.predictor_network(s_t)), tf.float32)
      metrics.update(self.opt(tape, loss, self.predictor_network))
      # print(loss)
    return metrics

  def _intr_reward(self, img):
    s_t = {'image': img}
    reward = tf.keras.metrics.mean_squared_error(tf.stop_gradient(self.target_network(s_t)),
                                                 tf.stop_gradient(self.predictor_network(s_t)))
    return reward



if __name__ == "__main__":
  ckptid = 'GEN-537'
  bufferid = 'GEN-537'
  checkpoint_name = 'variables_pretrained_env0.pkl'
  replay_buffer_name = 'train_replay_0'

  testbufferid = 'DRE-50'

  home = expanduser("~")
  basedir = f"{home}/logs/{ckptid}"
  buffer_basedir = f"{home}/logs/{bufferid}"
  expl_ckpt_basedir = f"{home}/logs/expl_ckpt/buffer{bufferid}"
  os.makedirs(expl_ckpt_basedir, exist_ok=True)

  # Make an agent? Or an RND module? (You can also get this to work with seek_color?)
   # (But has to use config?)
  config = {'expl_opt': {'opt': 'adam', 'lr': 3e-4, 'eps': 1e-5, 'clip': 100, 'wd': 1e-6},
            'rnd_recon_target_head': {'depth': 3, 'act': 'elu', 'kernels': [4, 4, 4, 4], 'keys': ['image'], 'outputsize': 512},
            'rnd_recon_predictor_head': {'depth': 5, 'act': 'elu', 'kernels': [4, 4, 4, 4], 'keys': ['image'], 'outputsize': 512},
            }
  expl = RND_raw(config)

  # Load in eps
  # which_eps = [[0], [1], [2], [3], [4]]
  loss = []
  which_eps = np.arange(1000)
  for batch_ep in tqdm(which_eps):
    eps = lu.load_eps(buffer_basedir, replay_buffer_name, batch_eps=[batch_ep])
    mets = expl._train_predictor_ep(eps[0], num_iter=10)
    loss.append(mets['RNDpred_loss'].numpy())

    # nt = eps[0]['action'].shape[0]
    # xyz = np.stack([ep['absolute_position_agent0'][:nt, :] for ep in eps], axis=0)
    # data = {key: np.vstack([np.expand_dims(ep[key].astype('float32'), axis=0)
    #                         for ep in eps])[:, :nt]
    #         for key in eps[0].keys()}
    # data = {key: tf.cast(data[key], dtype=data[key].dtype) for key in data.keys()}

  expl.save(os.path.join(expl_ckpt_basedir, f'{which_eps[-1]}_ckpt.pkl'))
  plt.plot(np.array(loss))
  plt.ylabel('RNDpred_loss')
  plt.xlabel('Ep #')
  plt.show()
  # Train agent



  # Load in test_eps
  intr_rew_train = expl._intr_reward(eps[0]['image'].astype('float32'))

  keys = ['yellow', 'magenta', 'noball']
  load_dir = f"{home}/logs/test_images_from_{testbufferid}/"
  intr_rews = {}
  for color in keys:
    load_imgs, imfs = iu.load_bmp(os.path.join(load_dir, color))
    intr_rew = expl._intr_reward(load_imgs.astype('float32'))
    intr_rews[color] = intr_rew

  plt.figure()
  for color in keys:
    plt.plot(intr_rews[color])
  plt.plot(intr_rew_train, 'k')
  keys.append('train_set')
  plt.legend(keys)
  plt.xlim(0, 200)
  plt.xlabel('Image #')
  plt.ylabel('Intr rew')
  plt.show()

  print('done')







