"""Launch dreamer without having to use command line args."""
import argparse
import os
import pathlib
import warnings
import neptune.new as neptune
import tensorflow as tf
import shutil

from dreamerv2 import train
from training.utils.neptune_utils import NeptuneLogger
from dreamerv2.utils import parse_flags

print(os.getcwd())


def main(logdir=None, do_neptune=True, do_debug=False):
    parser = argparse.ArgumentParser(description='Specify training arguments.')
    parser.add_argument('--logdir', dest='logdir', default='/dev/null',
                        help='path to logdir')
    parser.add_argument('--env_sequence',
                        default='env_sequence_ballrun.yaml',
                        help='path to logdir')
    args = parser.parse_args()
    if args.logdir != '/dev/null':
        logdir = args.logdir

    if logdir is None:
        logdir = args.logdir

    flags = [
        '--configs', 'defaults', 'dmc',
        '--dataset.batch', '10',
        '--train_every', '5',
        '--env_sequence', args.env_sequence,
        '--delete_old_trajectories', 'True',
        '--replay_size', '5e5',
        '--ckpt_each_eval', '1',
        '--seed', '1',

        '--time_limit', '1e9', # Nonepisodic (freerange)
        # '--time_limit', '1e4', # Episodic. Note: this is not divided by config.action_repeat
        # '--time_limit', '1000', # Episodic. Note: this is not divided by config.action_repeat
        '--save_freq', '500',  # Make this an integer fraction of eval_every
        '--eval_every', '50e4',  # Also, frequency of ckpt saveout
        '--log_every', '2e4',
        '--prefill', '5000',
        '--rssm.discrete', '0',
        # '--reset_position_freq', '1e3',   # This is the same scale as time_limit and steps
        # '--clear_buffer_at_step', '1e6', # Same scale as train steps
        # '--dataset.prioritize_temporal', 'True',

        '--eval_every', '2e3',  # Also, frequency of ckpt saveout
        '--log_every', '2e3',
        '--ckpt_load_path', 'DRA-83/variables_train_agent_envindex0_final_000250001.pkl',
        # '--ckpt_load_path', 'DRA-82/variables_train_agent_envindex0_final_000250001',
        '--inner_wm_train', '32',
        '--inner_explb_train', '32',


        # '--reset_state_freq', '1e3',
        # '--expl_noise', '10.0',

        # '--jit', 'False',
        # '--replay_buffer_source', 'GEN-EXAMPLE_EPS/train_replay_0',

        # '--expl_model_loss', 'true_reward', #'image',
        # '--pretrain_adapt', '1e3',  # Number of pretrain steps after loading checkpoint (check what is frozen)
        # '--pretrain_expl', '1e5', # Number of pretrain steps of expl_behavior
        # '--replay_buffer_source', 'GEN-2r/test_env_train_replay_1-1_new',
        # '--ckpt_load_path', 'GEN-533/variables_train_agent.pkl',
        # '--freeze_models', 'True',

       # '--ckpt_load_path', 'GEN-540/variables_train_agent_envindex0_005002500.pkl',
       # '--ckpt_load_path', 'DRE-193/variables_train_agent_envindex0_final_000502500.pkl',
       # '--ckpt_load_path', 'DRE-193/variables_train_agent_wm.pkl',

        # '--replay_buffer_source', 'DRE-193/train_replay_0',
       # '--reset_on_respawn', 'expl_actor_critic',
       # '--inner_wm_train', '8',
       # '--inner_explb_train', '8',
       #  '--intr_rew_train_every', '1e7',
       #  '--expl_policy_train_every', '1',
       #  '--pretrain_adapt', '0',  # Number of pretrain steps after loading checkpoint

        # '--freeze_rnd_pred', 'False',
       # '--rnd_pred_train_every', '10',
       # '--model_opt.lr', '6e-4',

    # '--replay_buffer_source', 'GEN-EXAMPLE_EPS/train_replay_0',
       # '--inner_explb_train', '8',

        # '--reset_on_respawn', 'expl_actor_critic', 'intr_reward_head',
        # '--reset_on_respawn', 'actor_critic', 'reward_head_test',
    ]
    flags.extend(['--logdir', logdir])
    if do_debug:
        print('DEBUG MODE')
        flags.extend(['--eval_every', '2e4'])
        flags.extend(['--jit', 'False'])
        flags.extend(['--dataset.batch', '3'])
        flags.extend(['--pretrain', '5'])

        # To make episodes happen faster
        flags.extend(['--prefill', '800'])
        flags.extend(['--time_limit', '1e8'])
        # flags.extend(['--time_limit', '100'])
        flags.extend(['--min_replay_episode_length', '5'])
        flags.extend(['--train_every', '25'])
        flags.extend(['--save_freq', '100'])
        # flags.extend(['--reset_state_freq', '50'])
        # flags.extend(['--clear_buffer_at_step', '800'])
        flags.extend(['--dataset.prioritize_temporal', 'True'])
        # flags.extend(['--dataset.priority_probs', "(0.6, 0.3, 0.1)"])
        # flags.extend(['--dataset.priority_chunks', "(30, 1200, -1)"])

    config, logdir = parse_flags(flags)

    which_gpu = 0
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[which_gpu], 'GPU')

    if do_neptune:
        if do_debug:
            project_name = f'Autonomous-Agents/sandbox'
            # tf.config.run_functions_eagerly(True)
        else:
            # project_name = f'Autonomous-Agents/GEN'
            # project_name = f'Autonomous-Agents/dre'
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

    train.main(logdir, config, neptune_logger)


if __name__ == '__main__':

    print("Launching from dreamer_launch.py")

    debug = False

    if debug:
        logdir = os.path.join(os.getenv('HOME'), 'logs', 'p2e_fiddle_dreamer-launch')
        try:
            shutil.rmtree(os.path.join(os.getenv('HOME'), 'logs', 'p2e_fiddle_dreamer-launch'))
            print('Deleted p2e_fiddle_dreamer-launch')
        except:
            print('Couldn\'t delete p2e_fiddle_dreamer-launch')

        main(logdir, do_neptune=True, do_debug=True)

    else:
        main()
