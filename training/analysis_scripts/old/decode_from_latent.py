"""
Make dataset consistent of latent state and behavioral variables.

Take in an episode trajectory, load a checkpointed model, compute latent state,
compare with behavioral variable.
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
from dreamerv2 import agent

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


if __name__ == "__main__":
  expid = 'GEN-304'
  basedir = f"/home/saal2/logs/{expid}"
  env_num = 0


  all_xyz = []
  all_logit = []
  all_deter = []

  for which_ep in range(2):
    # Load in trajectory from replay buffer
    traj_dir = f"{basedir}/train_replay_{env_num}/"
    traj_name = os.listdir(traj_dir)[which_ep]
    traj_timestamp = traj_name.split('-')[0]
    traj_path = f'{traj_dir}/{traj_name}'
    ep = np.load(traj_path)
    imgs = ep['image']
    xyz = ep['absolute_position_agent0']

    plt.figure()
    plt.plot(xyz[:, 0], xyz[:, 1], 'b.')
    plt.xlim([-15, 15])
    plt.ylim([-15, 15])
    plt.title(traj_name)
    plt.show()

    do_make_gif = False
    if do_make_gif:
      clip = ImageSequenceClip(list(imgs), fps=20)
      clip.write_gif(f'{basedir}/{traj_name}.gif', fps=20)


    # Load in behavior csv
    ### TODO: IF YOU ARE GOING TO USE df['step'] to align to traj -- REMEMBER TO DIVIDE BY 2
    fn = f'{basedir}/log_train_env{env_num}.csv'
    df = pd.read_csv(fn)
    t = get_closest_timestamp(df, traj_timestamp)
    df_ep = df[df['timestamp']==t]
    plt.figure()
    legend_str = []
    plt.plot(df['agent0_xloc'], df['agent0_yloc'], 'b.')
    plt.plot(df_ep['agent0_xloc'], df_ep['agent0_yloc'], 'r.')
    legend_str.append(f'agent0')
    plt.xlim([-15, 15])
    plt.ylim([-15, 15])
    plt.title(t)
    plt.show()


    ### Load in model...to compute latent state?

    # First load/initialize all the various stuff needed to init an agent
    for_ckpt_dir = f'{basedir}/for_ckpt'
    logdir = f'{basedir}/fiddle_dir'
    os.makedirs(logdir, exist_ok=True)

    with open(f'{for_ckpt_dir}/config_train_{env_num}.pkl', 'rb') as f:
      config = pickle.load(f)

    with open(f'{for_ckpt_dir}/action_space_train_{env_num}.pkl', 'rb') as f:
      action_space = pickle.load(f)

    step = elements.Counter(0)
    outputs = [
        elements.TerminalOutput(),
        elements.JSONLOutput(logdir),
        elements.TensorBoardOutput(logdir),
    ]
    logger = elements.Logger(step, outputs, multiplier=config.action_repeat)

    # Copy a few episodes over from main replay buffer so can init dataset
    os.makedirs(os.path.join(logdir, f'train_replay_{env_num}'), exist_ok=True)
    traj_dir = f"{basedir}/train_replay_{env_num}/"
    for traj_name in os.listdir(traj_dir)[:3]:
      shutil.copy(os.path.join(traj_dir, traj_name), os.path.join(logdir, f'train_replay_{env_num}', traj_name))

    replay_dir = pathlib.Path(f'{logdir}/train_replay_{env_num}')
    train_replay = common.Replay(replay_dir, config.replay_size, config)
    train_dataset = iter(train_replay.dataset(**config.dataset))

    agnt = agent.Agent(config, logger, action_space, step, train_dataset)
    agnt.load(f'{basedir}/variables_train_agent.pkl')
    print('Agent loaded from checkpoint!')

    ### Now, can I run a trajectory through the latent state encoder?
    # Make a 'pseudobatch' by just copying the episode six times
    data = {key: np.repeat(np.expand_dims(ep[key].astype('float32'), axis=0), 6, axis=0)
            for key in ['image', 'reward', 'discount', 'action']}

    # Generate latent state sequence for an entire trajectory
    nbatch = 2
    nt = -1
    wm = agnt.wm
    data = wm.preprocess(data)
    truth = data['image'][:nbatch, :nt] + 0.5
    embed = wm.encoder(data)
    states, _ = wm.rssm.observe(embed[:nbatch, :nt], data['action'][:nbatch, :nt])
    recon = wm.heads['image'](
      wm.rssm.get_feat(states)).mode()[:nbatch]
    recon += 0.5

    error = (recon - truth + 1) / 2
    video = tf.concat([truth, recon, error], 2)
    B, T, H, W, C = video.shape
    clip = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))

    # Now save out video of reconstruction (not open loop)
    clip = clip*255
    clip = clip.astype(np.uint8)
    clip = ImageSequenceClip(list(clip.numpy()), fps=20)
    clip.write_gif(f'{basedir}/recon.gif', fps=20)

    # TODO: Make a dataset of position vs. state (or vs embedding?) for a trajectory
    all_xyz.append(xyz[:nt])
    all_logit.append(states['logit'][0])
    all_deter.append(states['deter'][0])

    # xy = ep['absolute_position_agent0'][:nt]
    # logit = states['logit'][0]
    # deter = states['deter'][0]


    do_openl = False
    if do_openl:
      # Now build the latent state and make reconstruction (based on agnt.wm.video_pred(data))
      nbatch = 4
      wm = agnt.wm
      data = wm.preprocess(data)
      truth = data['image'][:nbatch] + 0.5
      embed = wm.encoder(data)
      states, _ = wm.rssm.observe(embed[:nbatch, :5], data['action'][:nbatch, :5])
      recon = wm.heads['image'](
        wm.rssm.get_feat(states)).mode()[:nbatch]

      init = {k: v[:, -1] for k, v in states.items()}
      prior = wm.rssm.imagine(data['action'][:nbatch, 5:], init)
      openl = wm.heads['image'](wm.rssm.get_feat(prior)).mode()
      model = tf.concat([recon[:, :5] + 0.5, openl + 0.5], 1)
      error = (model - truth + 1) / 2
      video = tf.concat([truth, model, error], 2)
      B, T, H, W, C = video.shape
      clip = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))

      # Now save out video
      clip = clip*255
      clip = clip.astype(np.uint8)
      clip = ImageSequenceClip(list(clip.numpy()), fps=20)
      clip.write_gif(f'{basedir}/pred.gif', fps=20)


    del agnt

  print('done')
