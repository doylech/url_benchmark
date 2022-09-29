import argparse
import os
from datetime import datetime

import imageio
import numpy as np
import matplotlib.pyplot as plt
import time

from dm_control import viewer
import common.envs
import training.utils.loading_utils as lu
import elements
from os.path import expanduser
import tensorflow as tf
from training.utils.diagnosis_utils import convert_matplotlib_fig_to_array, get_discounted_rewards
from dreamerv2.utils import parse_flags

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
tf.config.run_functions_eagerly(True)

def convert(value):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
        return value.astype(np.float32)
    elif np.issubdtype(value.dtype, np.signedinteger):
        return value.astype(np.int32)
    elif np.issubdtype(value.dtype, np.uint8):
        return value.astype(np.uint8)
    return value



def fiddle_with_env(config, num_frames=100, get_fps=False):


    bufferid = 'GEN-FIXED_ACTION'
    expid = 'GEN-dynamics-1-deep-test-noreset'  # 'GEN-1039-test'
    checkpoint_name = 'variables_test_agent_envindex1_000010000.pkl'
    log_names = ['towards_novel', 'towards_familiar']
    angle_offsets = [ np.pi/4.5, -np.pi/6]
    task = 'admc_sphero_multiagent_dynamics_test'

    # bufferid = 'GEN-FIXED_ACTION'
    # expid = 'GEN-3-1-test'  # 'GEN-1039-test'
    # checkpoint_name = 'variables_test_agent_envindex1_000100000.pkl'
    # log_names = ['towards_novel', 'towards_familiar']
    # angle_offsets = [ np.pi/4.5, -np.pi/6]
    # task = 'admc_sphero_multiagent_colorchange_test'

    ylim_intr_rew = [-4.5,-1.5]
    config = config.update(task=task)

    home = expanduser("~")
    basedir = f"{home}/gendreamer/logs/{expid}"
    buffer_basedir = f"{basedir}/{bufferid}"

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

    outputs = [
        elements.TensorBoardOutput(plot_dir),
    ]
    logger = elements.Logger(elements.Counter(0), outputs, multiplier=agnt.config.action_repeat)

    for angle_offset, log_name in zip(angle_offsets, log_names):
        env = common.envs.make_env(config, config.task, env_params={'randomize_spawn_rotation':False,
                                                                  'spawn_rotation_radians':{'0':3*np.pi/2 + angle_offset}})

        act = {k: np.array([-0.5, 0]) for k, v in env.action_space.spaces.items()}
        env.reset()
        acts = [act for _ in range(num_frames)]

        ep = []
        for i in range(num_frames):
            action = acts[i]
            ob, rew, done, info = env.step(action)
            disc = info.get('discount', np.array(1 - float(done)))
            tran = {**ob, **action, 'reward': rew, 'discount': disc, 'done': done}
            ep.append(tran)

        data = {k: convert([t[k] for t in ep])[None] for k in ep[0]}
        data = {key: tf.cast(data[key], dtype=data[key].dtype) for key in data.keys()}


        wm = agnt.wm
        data = wm.preprocess(data)
        embed = wm.encoder(data)
        post, _ = wm.rssm.observe(embed,
                                       data['action'])

        (feat, state,
         action, disc) = agnt.wm.imagine(agnt._expl_behavior.ac.actor,
                                         post,
                                         1)
        intr_reward = agnt._expl_behavior._intr_reward(feat, state, action)[1].numpy()
        # mem_reward = mem_reward[:, start_observing_at:end_imagining_at]

        discounted_mem_reward = get_discounted_rewards(intr_reward, [0,0.99])

        nrows = 2
        t = np.linspace(0, len(discounted_mem_reward[0])-1, 7).astype(np.int)
        plt.figure(figsize=(10, nrows * 2))
        plt.subplot2grid((nrows, len(t)), (0, 0), colspan=len(t))
        p2, = plt.plot(np.arange(len(discounted_mem_reward[0])), np.squeeze(discounted_mem_reward[0]), 'r')
        plt.ylabel('Intr Reward')
        plt.xlabel('steps')
        [plt.axvline(x, linestyle='--', color='k') for x in t]
        plt.ylim(ylim_intr_rew)

        B, T, H, W, C = data['image'].shape
        imgs_tf = data['image'].transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))  + 0.5
        imgs = imgs_tf.numpy()
        imgs = np.clip(imgs, 0, 1)
        imgs *= 255
        imgs = imgs.astype(np.int)
        logger.add({f'{checkpoint_name}_fixed_traj/{log_name}_video': imgs_tf})
        logger.write(fps=True)
        for i in range(len(t)):
            plt.subplot2grid((nrows, len(t)),
                             (1, i),
                             rowspan=2)
            plt.imshow(imgs[t[i]])
            plt.title(f't={t[i]}: R={discounted_mem_reward[0][t[i]]:0.3f}')
            plt.axis('off')
        plt.tight_layout()

        fig = plt.gcf()
        plot = convert_matplotlib_fig_to_array(fig)
        logger.image(f'{checkpoint_name}_fixed_traj/{log_name}', plot[None])
        logger.write()


def main():
    parser = argparse.ArgumentParser(description='Specify training arguments.')
    parser.add_argument('--logdir', dest='logdir', default='/dev/null',
                        help='path to logdir')
    args = parser.parse_args()

    flags = [
        '--configs', 'defaults', 'dmc', #'adapt',
        # '--task', 'admc_sphero_mazemultiagentInteract17',
        '--task', 'admc_sphero_multiagent_colorchange_test',
        # '--task', 'ddmc_walker_walk',
        '--egocentric_camera', 'True',
        ]
    flags.extend(['--logdir', args.logdir])

    config, logdir = parse_flags(flags)
    fiddle_with_env(config, num_frames=100)


if __name__ == "__main__":
    main()