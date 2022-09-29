import datetime
import io
import pathlib
import uuid
import os
import shutil

import numpy as np
import tensorflow as tf

class Replay:

  def __init__(self, directory, limit=None, config=None):
    directory.mkdir(parents=True, exist_ok=True)
    self._directory = directory
    self._limit = limit
    self._step = count_steps(directory)
    self._episodes = load_episodes(directory, limit)

    if config is not None:
      self._delete_old_trajectories = config.delete_old_trajectories
      self._min_episode_length = config.min_replay_episode_length
      try:
        self._clear_buffer_at_step = config.clear_buffer_at_step
      except:
        self._clear_buffer_at_step = -1
    else:
      self._delete_old_trajectories = False
      self._min_episode_length = 0
      self._clear_buffer_at_step = -1
      self._prioritize_temporal = False

    # Mem leak fix attempt
    #self.tfd = TfDataset()

  @property
  def total_steps(self):
    return self._step

  @property
  def num_episodes(self):
    return len(self._episodes)

  @property
  def num_transitions(self):
    return sum(self._length(ep) for ep in self._episodes.values())

  def add(self, episode):
    episode = {k:v for k,v in episode.items() if k!='info'}
    length = self._length(episode)
    update_steps(self._directory, length)
    self._step += length
    if self._clear_buffer_at_step > 0:
      if self._step > self._clear_buffer_at_step:
        print(f" !!! CLEARING BUFFER AT STEP: {self._step} (clear_buffer_at_step={self._clear_buffer_at_step}) !!! ")
        # shutil.rmtree(str(self._directory))
        # self._directory.mkdir(parents=True, exist_ok=True)
        keys = self._episodes.keys()
        keys = list(sorted(keys))
        n_to_delete = len(keys) - 1
        for ii in range(n_to_delete):
          key = keys[ii]
          os.remove(key)
          del self._episodes[key]
        self._clear_buffer_at_step = -1
    if length > self._min_episode_length:  # Skip saving short episodes to replay buffer
      if self._limit:
        total = 0
        for key, ep in reversed(sorted(
            self._episodes.items(), key=lambda x: x[0])):
          if total <= self._limit - length:
            total += self._length(ep)
          else:
            del self._episodes[key]
            if self._delete_old_trajectories:
              os.remove(key)
      filename = save_episodes(self._directory, [episode])[0]
      self._episodes[str(filename)] = episode
      return filename   ## IK added this

  def dataset(self, batch, length, oversample_ends, deterministic=False, prioritize_temporal=False):
    example = self._episodes[next(iter(self._episodes.keys()))]
    types = {k: v.dtype for k, v in example.items()}
    shapes = {k: (None,) + v.shape[1:] for k, v in example.items()}
    if prioritize_temporal:
      priority_probs = [1.0, 0.0]
      priority_chunks = [-1, -1]   # num episodes, [10 mins, 6 hours, all]
      # # priority_chunks = [30, 2160, -1]  # num episodes, [10 mins, 12 hours, all]
      # priority_chunks = [1, 3, -1]  # num episodes, [10 mins, 12 hours, all]

      print(priority_probs)
      print(priority_chunks)
      generator = lambda: sample_episodes_prioritized_temporal(
          self._episodes, length, oversample_ends,
          priority_probs=priority_probs, priority_chunks=priority_chunks)
    else:
      generator = lambda: sample_episodes(
          self._episodes, length, oversample_ends)
    if deterministic:
      generator = lambda: deterministic_episodes(
        self._episodes, length, oversample_ends)

    # FIXME: I think calling from_generator repeatedly could be causing a memory leak
    #   https://github.com/tensorflow/tensorflow/issues/37653
    dataset = tf.data.Dataset.from_generator(generator, types, shapes)

    #Mem leak fix attempt
    #dataset = self.tfd.from_generator(generator, types, shapes)

    dataset = dataset.batch(batch, drop_remainder=True)
    dataset = dataset.prefetch(10)
    return dataset

  def _length(self, episode):
    return len(episode['reward']) - 1


def save_episodes(directory, episodes):
  directory = pathlib.Path(directory).expanduser()
  directory.mkdir(parents=True, exist_ok=True)
  timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
  filenames = []
  for episode in episodes:
    identifier = str(uuid.uuid4().hex)
    length = len(episode['reward']) - 1
    filename = directory / f'{timestamp}-{identifier}-{length}.npz'
    with io.BytesIO() as f1:
      np.savez_compressed(f1, **episode)
      f1.seek(0)
      with filename.open('wb') as f2:
        f2.write(f1.read())
    filenames.append(filename)
  return filenames

def sample_episodes_prioritized_temporal(episodes, length=None, balance=False, seed=0,
                                         priority_probs=None, priority_chunks=None):
  random = np.random.RandomState(seed)

  # priority_probs = [0.6, 0.3, 0.1]
  # # priority_chunks = [30, 2160, -1]  # num episodes, [10 mins, 12 hours, all]
  # # priority_chunks = [30, 1200, -1]  # num episodes, [10 mins, 6 hours, all]
  # priority_chunks = [1, 3, -1]  # num episodes, [10 mins, 12 hours, all]

  while True:
    eps = list(episodes.keys())
    eps.sort(reverse=True)

    # First select which chunk it will come from
    which_chunk = np.where(random.multinomial(1, priority_probs))[0][0]
    ep = random.choice(eps[:priority_chunks[which_chunk]])
    # print(ep)
    episode = episodes[ep]

    if length:
      total = len(next(iter(episode.values())))
      available = total - length
      if available < 1:
        print(f'Skipped short episode of length {total}.')
        continue
      if balance:
        index = min(random.randint(0, total), available)
      else:
        index = int(random.randint(0, available + 1))
      episode = {k: v[index: index + length] for k, v in episode.items()}
    yield episode

def sample_episodes(episodes, length=None, balance=False, seed=0):
  random = np.random.RandomState(seed)
  while True:
    episode = random.choice(list(episodes.values()))
    if length:
      total = len(next(iter(episode.values())))
      available = total - length
      if available < 1:
        print(f'Skipped short episode of length {total}.')
        continue
      if balance:
        index = min(random.randint(0, total), available)
      else:
        index = int(random.randint(0, available + 1))
      episode = {k: v[index: index + length] for k, v in episode.items()}
    yield episode

def deterministic_episodes(episodes, length=None, balance=False, seed=0):
  while True:
    which_episodes = list(episodes.keys())
    which_episodes.sort()
    which_episode = which_episodes[0]
    print(which_episode)
    episode = episodes[which_episode]
    if length:
      index = 0
      episode = {k: v[index: index + length] for k, v in episode.items()}
    yield episode

def load_episodes(directory, limit=None):
  directory = pathlib.Path(directory).expanduser()
  episodes = {}
  total = 0
  for filename in reversed(sorted(directory.glob('*.npz'))):
    try:
      with filename.open('rb') as f:
        episode = np.load(f, allow_pickle=True)
        episode = {k: episode[k] for k in episode.keys()}
    except Exception as e:
      print(f'Could not load episode: {e}')
      continue
    episodes[str(filename)] = episode
    total += len(episode['reward']) - 1
    if limit and total >= limit:
      break
  return episodes


def count_steps(folder):
  f = str(folder) + '_steps.txt'
  if not os.path.isfile(f):
    np.savetxt(f, [0], fmt='%d')
  nsteps = int(np.loadtxt(f, ndmin=1)[0])
  return nsteps


def update_steps(folder, nsteps):
  f = str(folder) + '_steps.txt'
  if not os.path.isfile(f):
    np.savetxt(f, [0], fmt='%d')
  total_steps = nsteps + int(np.loadtxt(f, ndmin=1)[0])
  print(f'Updating stepcount to {total_steps}, {f}')
  np.savetxt(f, [total_steps], fmt='%d')