import os
import threading

import gym
import numpy as np

import dm_env
from dm_env import specs

import envs.playground.policies as pol
# from scipy.spatial.transform import Rotation as R

def make_env(config, name,  logging_params=None, env_params=None):
  """Wrappers for making various environments."""
  suite, task = name.split('_', 1)

  if suite == 'dmc':
    print(f'---> Using DeepMindControl: {task}')
    env = DMC(task, config.action_repeat, config.image_size)
    env = NormalizeAction(env)
  elif suite == 'admc':
    print(f'---> Using AdaptDeepMindControl: {task}')
    env = AdaptDMC(task, config.action_repeat, config.image_size,
                          aesthetic=config.aesthetic,
                          egocentric_camera=config.egocentric_camera,
                          env_params=env_params,
                          logging_params=logging_params,
                          control_timestep=config.control_timestep,
                          physics_timestep=config.physics_timestep,
                          reset_position_freq=config.reset_position_freq)
    env = NormalizeAction(env)
  elif suite == 'ddmc':
    print(f'---> Using DistractingDeepMindControl: {task}')
    env = DistractingDMC(task, config.action_repeat, config.image_size,
                                dynamic=config.dynamic_ddmc,
                                num_videos=config.num_videos_ddmc,
                                randomize_background=config.randomize_background_ddmc,
                                shuffle_background=config.shuffle_background_ddmc,
                                do_color_change=config.do_color_change_ddmc,
                                ground_plane_alpha=config.ground_plane_alpha_ddmc,
                                background_dataset_videos=config.background_dataset_videos_ddmc,
                                continuous_video_frames=config.continuous_video_frames_ddmc,
                                do_just_background=config.do_just_background_ddmc,
                                difficulty=config.difficulty_ddmc
                                )
    env = NormalizeAction(env)
  elif suite == 'procgen':
    print(f'----> Using procgen: {task}')
    env = ProcGen(task, config.action_repeat, config.image_size,
                         num_levels=config.pg_num_levels,
                         start_level=config.pg_start_level,
                         distribution_mode=config.pg_distribution_mode,
                         use_sequential_levels=config.pg_use_sequential_levels,
                         )
    env = OneHotAction(env)
  elif suite == 'atari':
    env = Atari(
        task, config.action_repeat, config.image_size, grayscale=config.grayscale,
        life_done=False, sticky_actions=True, all_actions=True)
    env = OneHotAction(env)
  else:
    raise NotImplementedError(suite)
  env = TimeLimit(env, config.time_limit)
  env = RewardObs(env)
  env = ResetObs(env)
  return env


class AdaptDMC:

  def __init__(self, name, action_repeat=1, size=(64, 64), camera=None, aesthetic='default',
               egocentric_camera=False, multiple_agents=False, env_params=None, logging_params=None,
               control_timestep=None, physics_timestep=None,
               reset_position_freq=None,
               ):
    domain, task = name.split('_', 1)
    if isinstance(domain, str):
      from envs.playground import suite
      self._env = suite.load(domain, task, environment_kwargs=dict(env_params=env_params,
                                                                   logging_params=logging_params,
                                                                   control_timestep=control_timestep,
                                                                   physics_timestep=physics_timestep,
                                                                   reset_position_freq=reset_position_freq))
      # self._env = suite.load(domain, task, environment_kwargs=dict(aesthetic=aesthetic))

    else:
      assert task is None
      self._env = domain()

    if 'multiagent' in task:
      print('----> APPLYING SpecifyPrimaryAgent WRAPPER.')
      primary_agent = self._env._task._walkers[self._env._task._primary_agent] # This is specified in the task.
      primary_agent_name = primary_agent._mjcf_root.model # Gets the name specified in the task, i.e. 'agent0'
      self._env = SpecifyPrimaryAgent(self._env, primary_agent_name, self._env.policies)

    self._action_repeat = action_repeat
    self._size = size
    if camera is None:
      camera = dict(quadruped=2, ant=2, humanoid=1, rodent=1).get(domain, 0)
      if 'maze' in task:
        # camera = dict(quadruped=2, ant=3, humanoid=1, rodent=1).get(domain, 0)
        camera = dict(quadruped=2, ant=3, humanoid=1, rodent=1, sphero=1).get(domain, 0)
      if egocentric_camera:
        camera = dict(ant=3, humanoid=3, rodent=4, sphero=2).get(domain, 0)
        if 'maze' in task:
          camera = dict(quadruped=2, ant=4, humanoid=1, rodent=5, sphero=2).get(domain, 0)
    self._camera = camera
    self._multiple_agents = multiple_agents

  @property
  def observation_space(self):
    spaces = {}
    for key, value in self._env.observation_spec().items():
      spaces[key] = gym.spaces.Box(
          -np.inf, np.inf, value.shape, dtype=np.float32)
    spaces['image'] = gym.spaces.Box(
        0, 255, self._size + (3,), dtype=np.uint8)
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    spec = self._env.action_spec()
    if isinstance(spec, list):
      spec = spec[0]
    action = gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    return gym.spaces.Dict({'action': action})

  def step(self, action):
    action = action['action']
    assert np.isfinite(action).all(), action
    reward = 0
    for _ in range(self._action_repeat):
      time_step = self._env.step(action)
      reward += time_step.reward or 0
      if time_step.last():
        break
    obs = dict(time_step.observation)
    obs['image'] = self.render()
    done = time_step.last()
    collision_tracker = self._env._base_env.exploration_tracker.collision_tracker
    attention_tracker = self._env._base_env.exploration_tracker.attention_tracker
    info = {'discount': np.array(time_step.discount, np.float32),
            'collision_tracker':collision_tracker, 'attention_tracker': attention_tracker}
    return obs, reward, done, info

  def reset(self):
    time_step = self._env.reset()
    if self._multiple_agents:
      obs = dict(time_step.observation[0])
    else:
      obs = dict(time_step.observation)
    obs['image'] = self.render()
    return obs

  def render(self, *args, **kwargs):
    if kwargs.get('mode', 'rgb_array') != 'rgb_array':
      raise ValueError("Only render mode 'rgb_array' is supported.")
    return self._env.physics.render(*self._size, camera_id=self._camera)


class DistractingDMC:
  def __init__(self, name, action_repeat=1, size=(64, 64), camera=None, dynamic=True,
               num_videos=1, randomize_background=None, shuffle_background=0,
               do_color_change=False, ground_plane_alpha=1.0, background_dataset_videos=None,
               continuous_video_frames=False, do_just_background=True, difficulty='easy'):
    """

    :param name:
    :param action_repeat:
    :param size:
    :param camera:
    :param dynamic: bool. Whether to use dynamic videos as opposed to static images.
    :param num_videos: int. How many different videos to use.
    :param randomize_background: 0 or 1. Randomize within each video, but videos keep same sequence ordering (i.e. are predictable).
    :param shuffle_background: 0 or 1. Shuffle all videos together, should be no predictability in background.
    """
    if background_dataset_videos is None:
      print('Using default for background video dataset: the DAVIS training set')
      background_dataset_videos = 'train'

    print(f'---> Loading distracting dreamer: {name}')
    domain, task = name.split('_', 1)
    if domain == 'cup':  # Only domain with multiple words.
      domain = 'ball_in_cup'
    if isinstance(domain, str):
      from envs.distracting_control import suite
      self._env = suite.load(domain, task, difficulty=difficulty,
                             pixels_only=False, do_just_background=do_just_background,
                             do_color_change=do_color_change,
                             background_dataset_videos=background_dataset_videos,
                             background_kwargs=dict(num_videos=num_videos,
                                                    dynamic=dynamic,
                                                    randomize_background=randomize_background,
                                                    shuffle_buffer_size=shuffle_background*500,
                                                    seed=1,
                                                    ground_plane_alpha=ground_plane_alpha,
                                                    continuous_video_frames=continuous_video_frames))
    else:
      assert task is None
      self._env = domain()
    self._action_repeat = action_repeat
    self._size = size
    if camera is None:
      camera = dict(quadruped=2).get(domain, 0)
    self._camera = camera

  @property
  def observation_space(self):
    spaces = {}
    for key, value in self._env.observation_spec().items():
      spaces[key] = gym.spaces.Box(
          -np.inf, np.inf, value.shape, dtype=np.float32)
    spaces['image'] = gym.spaces.Box(
        0, 255, self._size + (3,), dtype=np.uint8)
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    spec = self._env.action_spec()
    action = gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)
    return gym.spaces.Dict({'action': action})

  def step(self, action):
    action = action['action']
    assert np.isfinite(action).all(), action
    reward = 0
    for _ in range(self._action_repeat):
      time_step = self._env.step(action)
      reward += time_step.reward or 0
      if time_step.last():
        break
    obs = dict(time_step.observation)
    obs['image'] = self.render()
    del obs['pixels']  # Remove 'pixels' to match original format.
    done = time_step.last()
    info = {'discount': np.array(time_step.discount, np.float32)}
    return obs, reward, done, info

  def reset(self):
    time_step = self._env.reset()
    obs = dict(time_step.observation)
    obs['image'] = self.render()
    del obs['pixels']  # Remove 'pixels' to match original format.
    return obs

  def render(self, *args, **kwargs):
    if kwargs.get('mode', 'rgb_array') != 'rgb_array':
      raise ValueError("Only render mode 'rgb_array' is supported.")
    return self._env.physics.render(*self._size, camera_id=self._camera)


class DMC:

  def __init__(self, name, action_repeat=1, size=(64, 64), camera=None):
    os.environ['MUJOCO_GL'] = 'egl'
    domain, task = name.split('_', 1)
    if domain == 'cup':  # Only domain with multiple words.
      domain = 'ball_in_cup'
    if isinstance(domain, str):
      from dm_control import suite
      self._env = suite.load(domain, task)
    else:
      assert task is None
      self._env = domain()
    self._action_repeat = action_repeat
    self._size = size
    if camera is None:
      camera = dict(quadruped=2).get(domain, 0)
    self._camera = camera

  @property
  def observation_space(self):
    spaces = {}
    for key, value in self._env.observation_spec().items():
      spaces[key] = gym.spaces.Box(
          -np.inf, np.inf, value.shape, dtype=np.float32)
    spaces['image'] = gym.spaces.Box(
        0, 255, self._size + (3,), dtype=np.uint8)
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    spec = self._env.action_spec()
    action = gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)
    return gym.spaces.Dict({'action': action})

  def step(self, action):
    action = action['action']
    assert np.isfinite(action).all(), action
    reward = 0
    for _ in range(self._action_repeat):
      time_step = self._env.step(action)
      reward += time_step.reward or 0
      if time_step.last():
        break
    obs = dict(time_step.observation)
    obs['image'] = self.render()
    done = time_step.last()
    info = {'discount': np.array(time_step.discount, np.float32)}
    return obs, reward, done, info

  def reset(self):
    time_step = self._env.reset()
    obs = dict(time_step.observation)
    obs['image'] = self.render()
    return obs

  def render(self, *args, **kwargs):
    if kwargs.get('mode', 'rgb_array') != 'rgb_array':
      raise ValueError("Only render mode 'rgb_array' is supported.")
    return self._env.physics.render(*self._size, camera_id=self._camera)

class ProcGen:

  LOCK = threading.Lock()

  def __init__(
      self, name, action_repeat, size,
          num_levels, start_level, distribution_mode, use_sequential_levels,
          grayscale=False,
          noops=30, life_done=False, sticky_actions=True, all_actions=False):
    assert size[0] == size[1]
    import gym
    with self.LOCK:
      env = gym.make(f"procgen:procgen-{name}-v0",
                     num_levels=num_levels, start_level=start_level,
                     distribution_mode=distribution_mode,
                     use_sequential_levels=use_sequential_levels)

    # Avoid unnecessary rendering in inner env.
    env._get_obs = lambda: None
    # Tell wrapper that the inner env has no action repeat.
    env.spec = gym.envs.registration.EnvSpec('NoFrameskip-v0')
    self._env = env
    self._grayscale = grayscale

  @property
  def observation_space(self):
    return gym.spaces.Dict({
        'image': self._env.observation_space,
        # 'ram': gym.spaces.Box(0, 255, (128,), np.uint8),
    })

  @property
  def action_space(self):
    return gym.spaces.Dict({'action': self._env.action_space})

  def close(self):
    return self._env.close()

  def reset(self):
    with self.LOCK:
      image = self._env.reset()
    if self._grayscale:
      image = image[..., None]
    obs = {'image': image,
           # 'ram': self._env.env._get_ram()
           }
    return obs

  def step(self, action):
    action = action['action']
    image, reward, done, info = self._env.step(action)
    if self._grayscale:
      image = image[..., None]
    obs = {'image': image,
           # 'ram': self._env.env._get_ram()
           }
    return obs, reward, done, info

  def render(self, mode):
    return self._env.render(mode)


class Atari:

  LOCK = threading.Lock()

  def __init__(
      self, name, action_repeat=4, size=(84, 84), grayscale=True, noops=30,
      life_done=False, sticky_actions=True, all_actions=False):
    assert size[0] == size[1]
    import gym.wrappers
    import gym.envs.atari
    if name == 'james_bond':
      name = 'jamesbond'
    with self.LOCK:
      env = gym.envs.atari.AtariEnv(
          game=name, obs_type='image', frameskip=1,
          repeat_action_probability=0.25 if sticky_actions else 0.0,
          full_action_space=all_actions)
    # Avoid unnecessary rendering in inner env.
    env._get_obs = lambda: None
    # Tell wrapper that the inner env has no action repeat.
    env.spec = gym.envs.registration.EnvSpec('NoFrameskip-v0')
    env = gym.wrappers.AtariPreprocessing(
        env, noops, action_repeat, size[0], life_done, grayscale)
    self._env = env
    self._grayscale = grayscale

  @property
  def observation_space(self):
    return gym.spaces.Dict({
        'image': self._env.observation_space,
        'ram': gym.spaces.Box(0, 255, (128,), np.uint8),
    })

  @property
  def action_space(self):
    return gym.spaces.Dict({'action': self._env.action_space})

  def close(self):
    return self._env.close()

  def reset(self):
    with self.LOCK:
      image = self._env.reset()
    if self._grayscale:
      image = image[..., None]
    obs = {'image': image, 'ram': self._env.env._get_ram()}
    return obs

  def step(self, action):
    action = action['action']
    image, reward, done, info = self._env.step(action)
    if self._grayscale:
      image = image[..., None]
    obs = {'image': image, 'ram': self._env.env._get_ram()}
    return obs, reward, done, info

  def render(self, mode):
    return self._env.render(mode)


class Dummy:

  def __init__(self):
    pass

  @property
  def observation_space(self):
    image = gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8)
    return gym.spaces.Dict({'image': image})

  @property
  def action_space(self):
    action = gym.spaces.Box(-1, 1, (6,), dtype=np.float32)
    return gym.spaces.Dict({'action': action})

  def step(self, action):
    obs = {'image': np.zeros((64, 64, 3))}
    reward = 0.0
    done = False
    info = {}
    return obs, reward, done, info

  def reset(self):
    obs = {'image': np.zeros((64, 64, 3))}
    return obs


class TimeLimit:

  def __init__(self, env, duration):
    self._env = env
    self._duration = duration
    self._step = None

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    assert self._step is not None, 'Must reset environment.'
    obs, reward, done, info = self._env.step(action)
    self._step += 1
    info["TimeLimit.truncated"] = False # Added this for compatibility with gym, stablebaselines3
    if self._step >= self._duration:
      done = True
      if 'discount' not in info:
        info['discount'] = np.array(1.0).astype(np.float32)
      info["TimeLimit.truncated"] = True # Added this for compatibility with gym, stablebaselines3
      self._step = None
    return obs, reward, done, info

  def reset(self):
    self._step = 0
    return self._env.reset()


class NormalizeAction:

  def __init__(self, env, key='action'):
    self._env = env
    self._key = key
    space = env.action_space[key]
    self._mask = np.isfinite(space.low) & np.isfinite(space.high)
    self._low = np.where(self._mask, space.low, -1)
    self._high = np.where(self._mask, space.high, 1)

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def action_space(self):
    low = np.where(self._mask, -np.ones_like(self._low), self._low)
    high = np.where(self._mask, np.ones_like(self._low), self._high)
    space = gym.spaces.Box(low, high, dtype=np.float32)
    return gym.spaces.Dict({**self._env.action_space.spaces, self._key: space})

  def step(self, action):
    orig = (action[self._key] + 1) / 2 * (self._high - self._low) + self._low
    orig = np.where(self._mask, orig, action[self._key])
    return self._env.step({**action, self._key: orig})


class OneHotAction:

  def __init__(self, env, key='action'):
    assert isinstance(env.action_space[key], gym.spaces.Discrete)
    self._env = env
    self._key = key
    self._random = np.random.RandomState()

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def action_space(self):
    shape = (self._env.action_space[self._key].n,)
    space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
    space.sample = self._sample_action
    space.n = shape[0]
    return gym.spaces.Dict({**self._env.action_space.spaces, self._key: space})

  def step(self, action):
    index = np.argmax(action[self._key]).astype(int)
    reference = np.zeros_like(action[self._key])
    reference[index] = 1
    if not np.allclose(reference, action[self._key]):
      raise ValueError(f'Invalid one-hot action:\n{action}')
    return self._env.step({**action, self._key: index})

  def reset(self):
    return self._env.reset()

  def _sample_action(self):
    actions = self._env.action_space.n
    index = self._random.randint(0, actions)
    reference = np.zeros(actions, dtype=np.float32)
    reference[index] = 1.0
    return reference


class RewardObs:

  def __init__(self, env, key='reward'):
    assert key not in env.observation_space.spaces
    self._env = env
    self._key = key

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    space = gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32)
    return gym.spaces.Dict({
        **self._env.observation_space.spaces, self._key: space})

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs['reward'] = reward
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    obs['reward'] = 0.0
    return obs


class ResetObs:

  def __init__(self, env, key='reset'):
    assert key not in env.observation_space.spaces
    self._env = env
    self._key = key

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    space = gym.spaces.Box(0, 1, (), dtype=bool)
    return gym.spaces.Dict({
        **self._env.observation_space.spaces, self._key: space})

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs['reset'] = np.array(False, bool)
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    obs['reset'] = np.array(True, bool)
    return obs


class SpecifyPrimaryAgent:
  """Specify a primary agent and policies to control all
  other agents in the env, forming a facade such that
  higher level methods think there is only one agent."""

  def __init__(self, env, primary_agent, policies=None, just_primary_agent_obs=True, base_env=None):
    """
    :param primary_agent: Name of the agent as specified in by the env, i.e. 'agent0'
    :param policies: dict with policy function handle for each non-primary agent. If contains
                     just a single policy function handle, then that is applied to all agents.
    :param just_primary_agent_obs: bool. Whether to make visible observations of all agents,
          as opposed to just the primary agent.
    :param base_env: The base environment, before any wrappers are applied. This is in case
          any wrappers have been applied prior to calling this wrapper.
    """
    self._env = env
    self._primary_agent = primary_agent
    self._just_primary_agent_obs = just_primary_agent_obs
    self._policies = policies
    self._step_number = 0

    if base_env is None:
      self._base_env = env
    else:
      self._base_env = base_env

    self._prev_full_time_step = None   # Previous time step with info for all agents

    # Extract action indices for each agent
    action_names = self._base_env.action_spec().name.split('\t')
    agent_names = list(set(([x.split('/')[0] for x in action_names])))
    agent_inds = {}
    for name in agent_names:
      agent_inds[name] = np.where([name in x for x in action_names])[0]     # TODO: Is this a potential bug?
    self._agent_action_inds = agent_inds


  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    spec = self.observation_spec()
    spaces = {}
    for key, value in spec.items():
      spaces[key] = gym.spaces.Box(
        -np.inf, np.inf, value.shape, dtype=np.float32)

    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    spec = self.action_spec()
    space = gym.space.Box(
      spec.minimum, spec.maximum, dtype=np.float32)

    return space

  def _trim_time_step(self, full_time_step):
    """
    If self._just_primary_agent_obs == True,
    then removes every entry from the time step that is not
    related to the primary agent.

    :param full_time_step: Time step containing observations for all agents.
    :return: time_step: Time step with observations potentially just for primary agent.
    """
    if self._just_primary_agent_obs:
      full_obs = full_time_step.observation
      obs = dict()
      for key, value in full_obs.items():
        if self._primary_agent in key:
          obs[key] = value
      time_step = dm_env.TimeStep(
        step_type=full_time_step.step_type,
        reward=full_time_step.reward,
        discount=full_time_step.discount,
        observation=obs)
    else:
      time_step = full_time_step

    return time_step

  def step(self, action):
    # If there is a 'policy' for the primary_agent, which may just be for resetting its position, apply that
    if self._primary_agent in self._policies.keys():
      spec = self.agent_action_spec(self._primary_agent)()
      self._policies[self._primary_agent](self._prev_full_time_step, self.physics, self._primary_agent, spec, self._step_number)
      # print('APPLYING PRIMARY_AGENT POLICY')

    # First, get actions for all non-primary agents
    agent_actions = dict()
    for agent, inds in self._agent_action_inds.items():
      if not agent == self._primary_agent:
        spec = self.agent_action_spec(agent)()
        if self._policies is None:
          agent_actions[agent] = pol.random(self._prev_full_time_step, self.physics,agent, spec, self._step_number)
        else:
          if isinstance(self._policies, dict):
            agent_actions[agent] = self._policies[agent](self._prev_full_time_step, self.physics,agent, spec, self._step_number)
          else:
            agent_actions[agent] = self._policies(self._prev_full_time_step, self.physics, agent, spec, self._step_number)

    command = np.zeros(self._base_env.action_spec().shape)
    for agent in self._agent_action_inds.keys():
      if not agent == self._primary_agent:
        command[self._agent_action_inds[agent]] = agent_actions[agent]

    # Include specified action for primary agent.
    command[self._agent_action_inds[self._primary_agent]] = action

    # Take simulation step, and record observation.
    full_time_step = self._base_env.step(command)
    self._prev_full_time_step = full_time_step

    # Now trim entries in time_step so that the observation matches spec
    time_step = self._trim_time_step(full_time_step)         # TODO: Is this a potential bug?

    self._step_number += 1

    return time_step

  def reset(self):
    full_time_step = self._env.reset()
    self._prev_full_time_step = full_time_step
    time_step = self._trim_time_step(full_time_step)
    self._step_number = 0

    return time_step

  def agent_action_spec(self, agent):
    """Clone the action_spec, but only including
    agent-specific actions."""
    full_spec = self._env.action_spec()
    inds = self._agent_action_inds[agent]
    spec = specs.BoundedArray(shape=inds.shape,
                              dtype=full_spec.dtype,
                              minimum=full_spec.minimum[inds],
                              maximum=full_spec.maximum[inds],
                              name='\t'.join([full_spec.name.split('\t')[i] for i in inds]))
    return lambda: spec

  def agent_observation_spec(self, agent):
    """Clone the observation_spec, but only including
    agent-specific observations."""
    full_spec = self._env.observation_spec()
    spec = type(full_spec)()
    for key, value in full_spec.items():
      if agent in key:
        spec[key] = value

    return lambda: spec

  def full_action_spec(self):
    return self._env.action_spec

  @property
  def action_spec(self):
    """Only makes visible the spec for the primary agent.
    If want to see full spec for all agents, use full_action_spec()."""
    return self.agent_action_spec(self._primary_agent)

  @property
  def observation_spec(self):
    if self._just_primary_agent_obs:
      return self.agent_observation_spec(self._primary_agent)
    else:
      return self._env.observation_spec

  @property
  def physics(self):
    return self._env.physics