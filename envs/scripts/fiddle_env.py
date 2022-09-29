import argparse
import os
from datetime import datetime

import imageio
import numpy as np
import matplotlib.pyplot as plt
import time

from dm_control import viewer
import common.envs

from dreamerv2.utils import parse_flags


def fiddle_with_env(config, interactive=True, get_fps=False, do_random = True):
    which_envs = 1

    if which_envs == 1:
        # env = common.envs.make_env(config, 'train')
        #logging_params = {'logger': None,
        #                  'grid_density': 20,
        #                  'episode_length': None,
        #                  'env_raw_output_file_name': 'log_' + datetime.now().strftime("%Y%m%d%H%M%S") + '.csv',
        #                  'record_every_k_timesteps': 20,
        #                  'episodes_for_summary_metrics': 2}

        #env = common.envs.make_env(config, config.task, logging_params)
        env = common.envs.make_env(config, config.task)
        action_spec = env._env._env._env._env._env.action_spec()

        if isinstance(action_spec, list):
            action_spec = action_spec[0]
    env.reset()

    if get_fps:
        interactive = False

    if interactive:
        env = env._env._env._env._env._env
        if do_random:
            # Define a uniform random policy.
            def policy(time_step):
                del time_step  # Unused.
                return np.random.uniform(low=action_spec.minimum,
                                         high=action_spec.maximum,
                                         size=action_spec.shape)
        else:
            def policy(time_step):
                del time_step
                action = np.zeros(action_spec.shape)
                # action[0] = 0.01
                return action

        # Launch the viewer GUI application.
        viewer.launch(env, policy=policy)

    else:
        acts = env.action_space

        base_env = env._env._env._env._env._env
        for i in range(base_env.physics.model.ncam):  # 20
            plt.subplot(np.ceil(base_env.physics.model.ncam/4), 4, i + 1)
            plt.imshow(base_env.physics.render(camera_id=i))
            plt.axis('off')
        plt.show()

        # Get the camera image? Egocentric? Profile?
        # See what the output of an env.step is.
        if get_fps:
            n = int(1e4)
            t0 = time.time()
            for i in range(n):
                action = acts.sample()
                results = [env.step(action)]
                obs, _, done = zip(*[p[:3] for p in results])
                if done[0]:
                    plt.figure()
                    plt.imshow(obs[0]['image']), plt.show()
                    plt.title(str(i))
                    print(i)
                    env.reset()
            t1 = time.time()
            fps = n/(t1 - t0)
            print(f'FPS: {fps}')
        else:
            action = acts.sample()
            results = [env.step(action)]
            obs, _, done = zip(*[p[:3] for p in results])
            obs = list(obs)
            plt.imshow(obs[0]['image']), plt.show()
        # Save out frames from a few episodes
        num_ep, num_frames = 2, 10

        for j in range(num_ep):
            for i in range(num_frames):
                action = acts.sample()
                results = [env.step(action)]
                obs, _, done = zip(*[p[:3] for p in results])
                obs = list(obs)
                plt.imshow(obs[0]['image']), plt.savefig(f'{i}.png')

            images = []
            for filename in [f'{i}.png' for i in range(num_frames)]:
                images.append(imageio.imread(filename))
                os.remove(filename)
            imageio.mimsave(f'{j}.gif', images, duration=0.1)

            # Reset
            env.reset()


def main():
    parser = argparse.ArgumentParser(description='Specify training arguments.')
    parser.add_argument('--logdir', dest='logdir', default='/dev/null',
                        help='path to logdir')
    args = parser.parse_args()

    flags = [
        '--configs', 'defaults', 'dmc', #'adapt',
        '--task', 'admc_sphero_multiagent_novel_objects_step1_objects',
        # '--task', 'admc_sphero_multiagent_novel_objects_test',
        # '--task', 'admc_sphero_multiagent_dense_goal',
        # '--reset_position_freq', '100',
        # '--task', 'ddmc_walker_walk',
        '--egocentric_camera', 'True',
        ]
    flags.extend(['--logdir', args.logdir])

    config, logdir = parse_flags(flags)
    fiddle_with_env(config, interactive=True, get_fps=False, do_random=True)


if __name__ == "__main__":
    main()