import math

import PIL.Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
import scipy.signal
import seaborn as sns
from PIL import Image
import glob

def main():

  user = 'ikauvar'

  if user == 'ikauvar':
    local_user = 'saal2'
    id = 'GEN-462'
    remote = 't4-tf-8'
    gcp_zone = 'us-central1-a'
    env_nums = [0]
    force_download = 1
    gcloud_path = '/home/saal2/google-cloud-sdk/bin/gcloud'
  elif user == 'cd':
    local_user = 'cd'
    id = 'GEN-303'
    remote = 'cd-gendreamer-1'
    gcp_zone = 'us-central1-a'
    env_nums = [0]
    force_download = True
    gcloud_path = '/snap/bin/gcloud'


  all_df = None
  changepoints = []
  for env_num in env_nums:
    fn = f'/home/{local_user}/logs/log_{id}_train_env{env_num}.csv'
    if force_download or not os.path.exists(fn):
      # First pull down the file
      print(f'Fetching log file from {remote}.')
      p = subprocess.Popen([gcloud_path, 'compute', 'scp',
                              f'{user}@{remote}:/home/{user}/logs/{id}/log_train_env{env_num}.csv',
                              fn,
                              '--recurse', '--zone', gcp_zone,
                              '--project', 'hai-gcp-artificial'],
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
      (output, err) = p.communicate()
      p_state = p.wait()
      print(output)

    # Second, analyze it
    df = pd.read_csv(fn)
    save_freq = np.diff(df['total_step'].to_numpy())[1]
    if all_df is None:
      all_df = pd.read_csv(fn)
    else:
      all_df = pd.concat((all_df, df))
    changepoints.append(all_df['total_step'].to_numpy()[-1])

    # labels = {'ball2': 'line',
    #           'ball3': 'circle',
    #           'ball4': 'still',
    #           'ball5': 'random'}
    # labels = {'ball2': 'light',
    #           'ball3': 'heavy',
    #           'ball4': 'heavy',
    #           'ball5': 'light'}
    # labels = {'ball2': '',
    #           'ball3': '',
    #           'ball4': '',
    #           'ball5': ''}
    labels = {'ball2': 'familiar',
              'ball3': 'familiar',
              'ball4': 'novel',
              'ball5': 'familiar'}
    # colors = {2: 'red',
    #           3: 'yellow',
    #           4: 'purple',
    #           5: 'blue'}
    labels = {
              'ball3': 'novel',
              'ball4': 'familiar',
              }
    colors = {
      3: 'orange',
      4: 'blue',
    }

    plt.figure(figsize=(8, 3))
    plt.subplot(1,2,1)
    legend_str = []
    plt.plot(df['agent0_xloc'], df['agent0_yloc'], 'k.',
             markersize=1, alpha=0.2)
    legend_str.append(f'agent0')
    for b in [2, 3, 4, 5]:
      try:
        plt.plot(df[f'ball{b}_xloc'], df[f'ball{b}_yloc'], '.',
                 markersize=2, color=colors[b], alpha=0.5)
        legend_str.append(f'ball{b}')
      except:
        pass
    plt.title(f'Position {id}')
    # plt.legend(legend_str)

    plt.subplot(1,2,2)
    legend_str = []
    for b in [2,3,4,5]:
      try:
        # plt.plot(scipy.signal.savgol_filter(df[f'collisions_ball{b}/shell'], 1001, 3))
        plt.plot(np.cumsum(df[f'collisions_ball{b}/shell']),
                 color=colors[b])
        legend_str.append(f'ball{b}:' + labels[f'ball{b}'])
      except:
        pass
    plt.legend(legend_str, bbox_to_anchor=[1.05, 1])
    plt.title(f'Cumulative collisions: env {env_num}')
    plt.tight_layout()
    plt.show()

    plt.figure()
    legend_str = []
    for b in [2,3,4,5]:
      try:
        plt.plot(df[f'collisions_ball{b}/shell'], color=colors[b])
        legend_str.append(f'ball{b}:' + labels[f'ball{b}'])
      except:
        pass
    plt.legend(legend_str)
    plt.show()

  # Plot two-ball collisions
  # Summarize all double collisions
  double_collisions = np.zeros(all_df['step'].to_numpy().shape)
  balls = [2, 3, 4, 5]
  for i1 in range(len(balls)):
    for i2 in range(i1+1, len(balls)):
      b1 = balls[i1]
      b2 = balls[i2]
      try:
        nc = all_df[f'collisions_ball{b1}/shell_ball{b2}/shell'].to_numpy()
      except:
        nc = 0
      double_collisions += nc
  plt.figure()
  plt.plot(np.cumsum(double_collisions))
  [plt.axvline(c/save_freq) for c in changepoints]
  plt.title(f'Double collisions {id}: env {env_num}')
  plt.show()


  plt.figure()
  legend_str = []
  for b in [2, 3, 4, 5]:
    try:
      plt.plot(np.cumsum(all_df[f'collisions_ball{b}/shell'].to_numpy()), color=colors[b])
      legend_str.append(f'ball{b}:' + labels[f'ball{b}'])
    except:
      pass
  plt.legend(legend_str)
  [plt.axvline(c/save_freq) for c in changepoints]
  plt.title(f'{id}: All cumulative collisions: envs {env_nums}')
  plt.show()

  plt.figure()
  plt.plot(np.cumsum(all_df['collisions_wall'].to_numpy()))
  [plt.axvline(c/save_freq) for c in changepoints]
  plt.title(f'{id}: All cumulative wall collisions: envs {env_nums}')
  plt.show()

  plt.figure()
  min_z = np.percentile(all_df['agent0_zloc'].to_numpy(), 0.05)
  plt.plot(all_df['agent0_zloc'].to_numpy()-min_z)
  plt.title(f'{id}: z_height, envs {env_nums}')
  plt.ylabel('height')
  plt.xlabel('step')
  plt.show()

  # Plot max zheight by episode
  plt.figure()
  min_z = np.percentile(all_df['agent0_zloc'].to_numpy(), 0.05)
  plt.plot(all_df.groupby('episode').max()['agent0_zloc'].to_numpy() - min_z)
  plt.title(f'{id}: z_height, envs {env_nums}')
  plt.xlabel('Episode')
  plt.ylabel('Max height in episode')
  plt.show()

  if 'agent0_steer_velocity' in all_df.keys():
    plt.figure()
    plt.plot(all_df['episode'].to_numpy(), all_df['agent0_steer_velocity'].to_numpy(), '.')
    plt.title(f'{id}: Steer velocity, envs {env_nums}')
    plt.xlabel('Episode')
    plt.ylabel('Rotational velocity')
    plt.show()


  if 'agent0_steer_velocity' in all_df.keys():
    ep_df = all_df.groupby(['episode']).mean()
    plt.figure(), plt.plot(np.abs(ep_df['agent0_steer_velocity']), '.')
    plt.ylabel('|Rotational velocity|')
    plt.xlabel('Episode')
    plt.title(f'{id}: Steer velocity, envs {env_nums}')
    plt.show()

  # Can I plot max height by episode? Might be a little more interpretable?


  ## TODO: Plot wall collisions
  print(all_df)
  ## TODO: Plot time spent in each quadrant

  try:
    write_explore_gif(all_df, id)
  except:
    print('didnt write gif')

  print('done')


def write_explore_gif(df, id):
  EPISODES_PER_CHUNK = 20
  WRITES_PER_EPISODE = 50
  GRID_DENSITY = 200
  IMAGE_SIZE = GRID_DENSITY * 10

  steps = EPISODES_PER_CHUNK * WRITES_PER_EPISODE
  image_directory = f'/home/cd/logs/gendreamer/{id}/'
  os.mkdir(image_directory)
  all_visits = np.zeros((GRID_DENSITY, GRID_DENSITY))

  num_chunks = math.ceil(len(df) / steps)

  for i in range(num_chunks):
    df_chunk = df[i*steps:(i+1)*steps]
    visits_this_chunk = map_x_y(df_chunk, GRID_DENSITY)
    all_visits += visits_this_chunk

    write_image(pixels=array_to_pixels(visits_this_chunk),
                name='chunk_count',
                image_num=i,
                image_directory=image_directory)

    write_image(pixels=array_to_pixels(all_visits),
                name='cumulative_count',
                image_num=i,
                image_directory=image_directory)

  for name in ['chunk_count', 'cumulative_count']:
    create_gif(image_directory + name + '_*.png',
               image_directory + 'GIF_' + name + '.gif',
               image_directory + 'SNAP_' + name + '.png')


def map_x_y(df, GRID_DENSITY):
  MAX_X, MAX_Y = 40, 40

  df['mapX'] = (df['agent0_xloc'] / MAX_X + 0.5) * GRID_DENSITY
  df['mapY'] = (df['agent0_yloc'] / MAX_Y + 0.5) * GRID_DENSITY

  df = df.round({'mapX': 0, 'mapY': 0})

  visits = np.zeros((GRID_DENSITY, GRID_DENSITY))

  for i in df.index:
    # Use (GRID_SIZE - Y, X) to align with visible orientation in fiddle_env
    visits[GRID_DENSITY-int(df['mapY'][i]), int(df['mapX'][i])] += 1

  return visits

def array_to_pixels(array: np.ndarray,
                    normalization: str = "log") -> np.ndarray:
  # Convert array to pixels
  if normalization == "scale":
    normed_array = array
    max = np.max(normed_array)
    pixels = 255 - (normed_array / max * 255)

  elif normalization == "log":
    normed_array = np.log(array + 1.0)
    pixels = 255 - np.minimum(normed_array * 25, 255)

  elif normalization == "root-raw-reward":
    normed_array = np.power(array, (1.0 / 3.0))
    pixels = 255 - np.minimum(normed_array * 200, 255)

  elif normalization == "linear-shift-value":
    normed_array = array + 20
    pixels = 255 - np.maximum(np.minimum(normed_array, 255), 0)

  elif normalization == "none":
    pixels = 255 - array

  else:
    raise ValueError('Unknown normalization: ' + normalization)

  # Convert type
  return pixels.astype('uint8')


def write_image(pixels: np.ndarray,
                name: str,
                image_num: int,
                image_directory: str) -> None:
  im = Image.fromarray(pixels)
  #im = im.resize((800,800), resample=PIL.Image.LANCZOS)
  im.save(image_directory + name + '_' + str(image_num).zfill(4) + '.png')

def create_gif(fp_in: str,
               fp_gif_out: str,
               fp_static_out: str) -> None:
  # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
  img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
  img.save(fp=fp_gif_out,
           format='GIF',
           append_images=imgs,
           save_all=True,
           duration=500,
           loop=0)

  img_static = Image.open(sorted(glob.glob(fp_in))[-1])
  img_static.save(fp=fp_static_out)


if __name__ == "__main__":
  main()