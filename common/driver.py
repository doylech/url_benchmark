import numpy as np
import datetime
import warnings
import elements


class Driver:

  def __init__(self, envs, logdir, save_freq=0, reset_state_freq=0, **kwargs):
    self._envs = envs
    self._kwargs = kwargs
    self._logdir = logdir
    self._on_steps = []
    self._on_resets = []
    self._on_episodes = []
    self._actspaces = [env.action_space.spaces for env in envs]
    self._save_freq = save_freq
    self.reset_state_freq = reset_state_freq
    if save_freq == 0:
      self.should_save = lambda step: False
    else:
      self.should_save = lambda step: (step+1) % save_freq == 0
    if reset_state_freq == 0:
      self.should_reset_state = lambda step: False
    else:
      self.should_reset_state = lambda step: (step+1) % reset_state_freq == 0

    self.reset()

  def on_step(self, callback):
    self._on_steps.append(callback)

  def on_reset(self, callback):
    self._on_resets.append(callback)

  def on_episode(self, callback):
    self._on_episodes.append(callback)

  def reset(self):
    self._obs = [None] * len(self._envs)
    self._dones = [True] * len(self._envs)
    self._eps = [None] * len(self._envs)
    self._state = None

  def __call__(self, policy, steps=0, episodes=0, delete_error_eps=False, stop_if_error=False):
    step, episode = 0, 0
    while step < steps or episode < episodes:
      error_eps = False
      for i, done in enumerate(self._dones):
        if done:
          self._obs[i] = ob = self._envs[i].reset()
          act = {k: np.zeros(v.shape) for k, v in self._actspaces[i].items()}
          tran = {**ob, **act, 'reward': 0.0, 'discount': 1.0, 'done': False}
          [callback(tran, **self._kwargs) for callback in self._on_resets]
          self._eps[i] = [tran]
        if self.should_reset_state(step):
          self._obs[i]['reset'] = np.array(True, np.bool)
          print(f'Should be resetting state, step: {step}')
      obs = {k: np.stack([o[k] for o in self._obs]) for k in self._obs[0]}
      actions, self._state = policy(obs, self._state, **self._kwargs)
      actions = [
          {k: np.array(actions[k][i]) for k in actions}
          for i in range(len(self._envs))]
      assert len(actions) == len(self._envs)
      try:
        results = [e.step(a) for e, a in zip(self._envs, actions)]
      except:
        print('***ERROR IN DRIVER STEP')
        error_eps = True
        if not delete_error_eps:
          # Log error episode (even if episode has not ended).
          for i, (act, (ob, rew, done, info)) in enumerate(zip(actions, results)):
            obs = {k: self._convert(v) for k, v in obs.items()}
            disc = info.get('discount', np.array(1 - float(done)))
            tran = {**ob, **act, 'reward': rew, 'discount': disc, 'done': done}
            [callback(tran, **self._kwargs) for callback in self._on_steps]
            self._eps[i].append(tran)
            ep = self._eps[i]
            ep = {k: self._convert([t[k] for t in ep]) for k in ep[0]}
            fnames = [callback(ep, **self._kwargs) for callback in self._on_episodes]
          print(f'***LOGGED TRAJECTORY FROM ERROR: {self._logdir}')
          warnings.warn(f'Failed driver step: {fnames[0]}')
          obs, _, dones = zip(*[p[:3] for p in results])
          self._obs = list(obs)
          self._dones = list(dones)
          episode += sum(dones)
          step += len(dones)
          # error_log = fnames[0].parents[1] / 'failed_eps_log.txt'
          with open(self._logdir / 'failed_eps_log.txt', 'a') as f:
            f.write(str(fnames) + '\n')
        else:
          with open(self._logdir / 'failed_eps_log.txt', 'a') as f:
            timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
            f.write(timestamp + '\n')
        if stop_if_error:
          raise

      if not error_eps:
        for i, (act, (ob, rew, done, info)) in enumerate(zip(actions, results)):
          obs = {k: self._convert(v) for k, v in obs.items()}
          disc = info.get('discount', np.array(1 - float(done)))
          tran = {**ob, **act, 'reward': rew, 'discount': disc, 'done': done}
          [callback(tran, **self._kwargs) for callback in self._on_steps]
          if self._eps[i] is None:
            self._eps[i] = [tran]
          else:
            self._eps[i].append(tran)
          if done or self.should_save(step):
            print(f'Save ep {step}')
            if self.should_save(step):
              ep = self._eps[i][:-1]
              self._eps[i] = [self._eps[i][-1]]
            else:
              ep = self._eps[i]
            ep = {k: self._convert([t[k] for t in ep]) for k in ep[0]}
            ep['info'] = info
            fnames = [callback(ep, **self._kwargs) for callback in self._on_episodes]
        obs, _, dones = zip(*[p[:3] for p in results])
        self._obs = list(obs)
        self._dones = list(dones)
        episode += sum(dones)
        step += len(dones)

  def _convert(self, value):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
      return value.astype(np.float32)
    elif np.issubdtype(value.dtype, np.signedinteger):
      return value.astype(np.int32)
    elif np.issubdtype(value.dtype, np.uint8):
      return value.astype(np.uint8)
    return value
