import argparse
import os
import pathlib
import warnings
import neptune.new as neptune
import shutil

import torch
from sb3 import sb3_train
from training.utils.neptune_utils import NeptuneLogger
from dreamerv2.utils import parse_flags

# TODO: Launch on server
# TODO: Checkpoint saving to log
# TODO: Tensorboard logging
# TODO: Run dense_reward, it should work.

def main(logdir=None, do_neptune=True, do_debug=False):
  parser = argparse.ArgumentParser(description='Specify training arguments.')
  parser.add_argument('--logdir', dest='logdir', default='/dev/null', help='path to logdir')
  parser.add_argument('--env_sequence',
                      default='admc_sphero_multiagent_dense_goal') ## Set environment here ##
  args = parser.parse_args()
  flags = [
    '--configs', 'defaults', 'dmc',
    '--dataset.batch', '10',
    '--train_every', '5',
    '--env_sequence', args.env_sequence,
    '--delete_old_trajectories', 'True',
    '--replay_size', '5e5',
    '--ckpt_each_eval', '1',
    '--seed', '1',

    '--time_limit', '2e3',  # Nonepisodic (freerange) Note: this is not divided by config.action_repeat
    # '--time_limit', '1e9',  # Nonepisodic (freerange) Note: this is not divided by config.action_repeat
    '--save_freq', '500',  # Make this an integer fraction of eval_every
    '--eval_every', '10e4',  # Also, frequency of ckpt saveout
    '--log_every', '2e4',
    '--prefill', '5000',
    '--rssm.discrete', '0',
  ]
  if args.logdir != '/dev/null':
    logdir = args.logdir
  if logdir is None:
    logdir = args.logdir
  flags.extend(['--logdir', logdir])
  if do_debug:
    print('DEBUG MODE')
    flags.extend(['--eval_every', '2e4'])
    flags.extend(['--jit', 'False'])
    flags.extend(['--dataset.batch', '3'])
    flags.extend(['--pretrain', '5'])

    # To make episodes happen faster
    flags.extend(['--prefill', '800'])
    flags.extend(['--time_limit', '1e3'])
    # flags.extend(['--time_limit', '100'])
    flags.extend(['--min_replay_episode_length', '5'])
    flags.extend(['--train_every', '25'])
    flags.extend(['--save_freq', '100'])
    # flags.extend(['--reset_state_freq', '50'])
    # flags.extend(['--clear_buffer_at_step', '800'])
    # flags.extend(['--dataset.prioritize_temporal', 'True'])
    # flags.extend(['--dataset.priority_probs', "(0.6, 0.3, 0.1)"])
    # flags.extend(['--dataset.priority_chunks', "(30, 1200, -1)"])

  config, logdir = parse_flags(flags)

  which_gpu = 0
  torch.cuda.device(which_gpu)
  print(f'{torch.cuda.get_device_name()}: {torch.cuda.current_device()}')

  if do_neptune:
    if do_debug:
      project_name = f'Autonomous-Agents/sandbox'
    else:
      project_name = f'Autonomous-Agents/dra'

    run = neptune.init(project=project_name)
    neptune_logger = NeptuneLogger(run)
  else:
    neptune_logger = NeptuneLogger(run=None)

  # If necessary, copy over replay from another expt
  if config.replay_buffer_source != '/dev/null':
    basedir = '/'.join(config.logdir.split('/')[:-1])
    train_buffer = config.replay_buffer_source
    eval_buffer = train_buffer.replace('train', 'eval')
    shutil.copytree(f'{basedir}/{train_buffer}', f'{logdir}/train_replay_0')
    shutil.copytree(f'{basedir}/{eval_buffer}', f'{logdir}/eval_replay_0')

  sb3_train.main(logdir, config, neptune_logger)


if __name__ == '__main__':

  print("Launching from sb3_launch.py")

  debug = False

  if debug:
    logdir = os.path.join(os.getenv('HOME'), 'logs', 'sb3_fiddle')
    try:
      shutil.rmtree(os.path.join(os.getenv('HOME'), 'logs', 'sb3_fiddle'))
      print('Deleted sb3_fiddle')
    except:
      print('Couldn\'t delete sb3_fiddle')

    main(logdir, do_neptune=True, do_debug=True)

  else:
    main()

