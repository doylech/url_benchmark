import elements
import gym
from gym.utils import seeding
import numpy as np
import cv2

import common.envs
from dreamerv2.utils import parse_flags


class Playground(gym.Env):
  metadata = {'render.modes': ['rgb_array', 'human']}

  def __init__(self, name, config=None):
    """
    A gym wrapper around a playground environment.
    :param name: Environment name
    :param config: either elements.Config format, or list of flags
    """
    if config is None or isinstance(config, list):
      all_flags = ['--configs', 'defaults', 'dmc', '--seed', '1']
      all_flags.extend(config)
      config, logdir = parse_flags(all_flags)

    self.env = common.envs.make_env(config, name)
    self.observation_space = self.env.observation_space['image']
    self.action_space = self.env.action_space['action']

  def step(self, action):
    obs, reward, done, info = self.env.step({'action': action})
    obs = np.ascontiguousarray(obs['image'])
    return obs, reward, done, info

  def reset(self):
    obs = np.ascontiguousarray(self.env.reset()['image'])
    return obs

  def render(self, mode='human'):
      img = self.env.render()
      if mode == 'rgb_array':
          return img
      elif mode == 'human':
        resized = cv2.resize(img, (400, 400), interpolation=cv2.INTER_AREA)
        resized = np.flip(resized, axis=2)
        cv2.imshow('image', resized.astype('uint8'))
        cv2.waitKey(10)

  def close(self):
    pass