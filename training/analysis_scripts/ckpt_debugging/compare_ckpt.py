"""
A test script to verify that loaded checkpoint matches actual model from training.

Before running this script, you must run dreamer_launch.py with debug=True,
AND, you must set DO_DEBUG_DETERMINISTIC = True in train.py
This should generate new logs in ~/logs/p2e_fiddle_dreamer-launch
that will then be loaded here.
"""

import training.utils.loading_utils as lu
import numpy as np
import pickle
import common
import pathlib
import tensorflow as tf

if __name__ == "__main__":
  logdir = '/home/saal2/logs/p2e_fiddle_dreamer-launch/'
  debug_ckpt_name =  'variables_train_agent_debug.pkl'
  # replay_buffer_name = 'fiddle_dir/train_replay_0'
  replay_buffer_name = 'train_replay_0'
  debug_deterministic = True

  # Load config and set numeric precision accordingly.
  # Must do this before load anything else.
  with open(f'{logdir}/for_ckpt/config_{debug_ckpt_name}', 'rb') as f:
    config = pickle.load(f)
  if config.precision == 16:
    from tensorflow.keras.mixed_precision import experimental as prec
    prec.set_policy(prec.Policy('mixed_float16'))

  from tensorflow.keras import mixed_precision as prec
  print(prec.global_policy().compute_dtype)

  with open(f'{logdir}/debug_start_latent.pkl', 'rb') as f:
    loaded_start = pickle.load(f)
  with open(f'{logdir}/debug_data.pkl', 'rb') as f:
    loaded_data = pickle.load(f)


  agnt_loaded = lu.load_agent(logdir,
                       checkpoint_name=debug_ckpt_name,
                       batch_size=3,
                       deterministic=debug_deterministic)
  wm_agnt_loaded = lu.load_agent(logdir,
                                 checkpoint_name=debug_ckpt_name,
                                 batch_size=3,
                                 deterministic=debug_deterministic)
  wm_agnt_loaded.wm.load(f'{logdir}/variables_train_wm_debug.pkl')


  do_load_data = True
  do_use_Replay = True
  if do_load_data:
    if do_use_Replay:
      replay = common.Replay(pathlib.Path(f'{logdir}/{replay_buffer_name}'),
                             config.replay_size, config)
      # dataset = iter(replay.dataset(**config.dataset, deterministic=True))
      length = config.dataset.length
      # length = 501
      dataset = iter(replay.dataset(batch=3, length=length,
                                    oversample_ends=False, deterministic=True))
      data = next(dataset)
    else:
      print('Loading episodes npz directly.')
      eps = lu.load_eps(logdir, replay_buffer_name, batch_size=1, start_ind=0)
      eps = [eps[0]]*3  ## Just copy the first episode three times to create a batch of duplicate episodes
      data = {key: np.vstack([np.expand_dims(ep[key].astype('float32'), axis=0) for ep in eps])[:,:config.dataset.length]
              for key in eps[0].keys()}
      data = {key: tf.cast(data[key], dtype=data[key].dtype) for key in data.keys()}
  else:
    data = next(iter(agnt_loaded._dataset))
  assert(np.max(data['image'][0, :].numpy() - data['image'][1, :].numpy()) == 0)
  assert(np.max(data['image'][0, :].numpy() - next(iter(agnt_loaded._dataset))['image'][1,:].numpy()) == 0)
  assert((data['absolute_position_agent0'].numpy() == loaded_data['absolute_position_agent0'].numpy()).all())
  assert((data['image'].numpy() == loaded_data['image'].numpy()).all())
  assert((data['action'].numpy() == loaded_data['action'].numpy()).all())

  # Compute 'start' latent state and compare it to saved state.
  with open(f'{logdir}/debug_preproc.pkl', 'rb') as f:
    loaded_preproc = pickle.load(f)
  with open(f'{logdir}/debug_embed.pkl', 'rb') as f:
    loaded_embed = pickle.load(f)

  computed_preproc = wm_agnt_loaded.wm.preprocess(data)
  assert((computed_preproc['image'].numpy() ==
        loaded_preproc['image'].numpy()).all())
  assert((computed_preproc['action'].numpy() ==
        loaded_preproc['action'].numpy()).all())
  computed_embed = wm_agnt_loaded.wm.encoder(computed_preproc)
  assert((computed_embed.numpy() == loaded_embed.numpy()).all())
  post, prior = wm_agnt_loaded.wm.rssm.observe(computed_embed,
                                               computed_preproc['action'],
                                               state=None,
                                               sample=(not debug_deterministic))
  start = post
  print('Computed start:')
  print(start['deter'][:, -1, :2])
  print('Loaded start:')
  print(loaded_start['deter'][:, -1, :2])
  assert ((start['deter'].numpy() == loaded_start['deter'].numpy()).all())

  model_loss, post, outs, metrics =  wm_agnt_loaded.wm.loss(data, state=None, sample=False)
  assert ((post['deter'].numpy() == loaded_start['deter'].numpy()).all())
  assert ((outs['post']['deter'].numpy() == loaded_start['deter'].numpy()).all())
  start = outs['post']

  (feat, state, ## state is: for t in horizon; for each starting latent state (from each timepoint of each batch); the latent state
   action, disc) = agnt_loaded.wm.imagine(agnt_loaded._expl_behavior.ac.actor,
                                          start,
                                          agnt_loaded.config.imag_horizon,
                                          deterministic=debug_deterministic)
  rewards = agnt_loaded._expl_behavior._intr_reward(feat, state, action)
  intr_reward = rewards[1]

  # Compare with saved reward
  with open(f'{logdir}/debug_intr_reward.npz', 'rb') as f:
    loaded_rewards = np.load(f)
    mets_intr_reward = loaded_rewards['mets_intr_reward']
    loaded_intr_reward = loaded_rewards['intr_reward']
  assert(intr_reward.mean() == mets_intr_reward)
  assert(intr_reward.mean() == loaded_intr_reward.mean())
  assert((intr_reward == loaded_intr_reward).all())
  print(f'Computed intr reward: {intr_reward.mean()}')

  print('\nSUCCESS!\n')