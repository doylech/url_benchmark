import math

import PIL.Image
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import subprocess
import os
import scipy.signal

from training.utils.diagnosis_utils import convert_matplotlib_fig_to_array
from PIL import Image
import glob
from os.path import expanduser
import elements
def main():
    env_num = 1
    exp_id ='GEN-dynamics-1-deep-test-wmeval-freeze' #'GEN-1039-test-closer2'
    # exp_id = 'GEN-tunneling-1'
    # exp_id = 'GEN-3-1-test-wmeval-freeze'
    # exp_id = 'GEN-mass-1-test'
    fn = f'/home/linqizhou/gendreamer/logs/{exp_id}/log_train_env{env_num}.csv'
    bufferid = f'GEN-EXAMPLE_EPS_{env_num}'
    home = expanduser("~")
    basedir = f"{home}/gendreamer/logs/{exp_id}"
    buffer_basedir = f"{basedir}/{bufferid}"

    plot_dir = f'{basedir}/plots'
    os.makedirs(plot_dir, exist_ok=True)


    outputs = [
        elements.TensorBoardOutput(plot_dir),
    ]
    logger = elements.Logger(elements.Counter(0), outputs, multiplier=2)

    changepoints = []
    all_df=None
    # Second, analyze it
    df = pd.read_csv(fn)
    # save_freq = np.diff(df['total_step'].to_numpy())[1]
    if all_df is None:
      all_df = pd.read_csv(fn)
    else:
      all_df = pd.concat((all_df, df))
    changepoints.append(all_df['total_step'].to_numpy()[-1])

    # plt.hist(df['init_rotation'][df['init_rotation'].notna() & (df['init_rotation']!='init_rotation')].to_numpy(), bins=10)
    # plt.xlim([-np.pi, np.pi])
    # plot = convert_matplotlib_fig_to_array(plt.gcf())
    # logger.image('agent initial rotation', plot[None])
    # plt.close()

    colors = ['blue', 'green','red','cyan','magenta','yellow','black']
    # plt.figure(figsize=(8, 3))
    # plt.subplot(1,2,1)
    legend_str = []

    plt.plot(df['agent0_xloc'].astype(float),
             df['agent0_yloc'].astype(float), 'k.', color='blue',
             markersize=1, alpha=0.2)
    legend_str.append(f'agent0')
    i=1
    for key in df.keys():
      if 'xloc' in key and 'agent0' not in key and 'target' not in key:
          xloc_key = yloc_key = key
          yloc_key = yloc_key[:-4] + 'y' + yloc_key[-3:]

          plt.plot(df[xloc_key].astype(float),
                   df[yloc_key].astype(float), '.',
                 markersize=2, color='orange' if 'novel' in key else colors[i], alpha=0.5)
          legend_str.append(key.split('_')[0])
          i+=1
    plt.legend(legend_str, bbox_to_anchor=[1.05, 1])
    plt.title(f'Position')
    plt.tight_layout()
    plot = convert_matplotlib_fig_to_array(plt.gcf())
    logger.image(f'{bufferid}/position', plot[None])
    plt.close()

    # plt.subplot(1,2,2)
    legend_str = []
    i = 0
    for key in df.keys():
      if 'collisions' in key and 'agent0' not in key and 'target' not in key and 'wall' not in key:
        try:

          # plt.plot(scipy.signal.savgol_filter(df[f'collisions_ball{b}/shell'], 1001, 3))
          plt.plot(np.cumsum(df[key].to_numpy().astype(float)),
                 color='orange' if 'novel' in key else colors[i])
          legend_str.append(key.split('_')[1])
          i+=1
        except:
          pass
    plt.legend(legend_str, bbox_to_anchor=[1.05, 1])
    plt.title(f'Cumulative collisions: env {env_num}')

    plt.tight_layout()
    plot = convert_matplotlib_fig_to_array(plt.gcf())
    logger.image(f'{bufferid}/collision', plot[None])
    plt.close()

    # plt.subplot(1,2,2)
    legend_str = []
    i=0
    for key in df.keys():
      if 'attention' in key and 'agent0' not in key and 'target' not in key:
        try:
          # plt.plot(scipy.signal.savgol_filter(df[f'collisions_ball{b}/shell'], 1001, 3))
          plt.plot(np.cumsum(df[key].to_numpy().astype(float)),
                 color='orange' if 'novel' in key else colors[i])
          legend_str.append(key.split('_')[1])
          i += 1
        except:
          pass
    plt.legend(legend_str, bbox_to_anchor=[1.05, 1])
    plt.title(f'Cumulative attention: env {env_num}')
    plt.tight_layout()
    plot = convert_matplotlib_fig_to_array(plt.gcf())
    logger.image(f'{bufferid}/attention', plot[None])
    plt.close()

    plt.figure()
    min_z = np.percentile(df['agent0_zloc'].astype(float), 0.05)
    plt.plot(df['agent0_zloc'].astype(float)- min_z)
    plt.title(f'z_height, env {env_num}')
    plt.ylabel('height')
    plt.xlabel('step')
    plot = convert_matplotlib_fig_to_array(plt.gcf())
    logger.image(f'{bufferid}/height', plot[None])
    plt.close()

    # Plot max zheight by episode
    plt.figure()
    min_z = np.percentile(df['agent0_zloc'].to_numpy().astype(float), 0.05)
    plt.plot(df.groupby('episode').max()['agent0_zloc'].astype(float) - min_z)
    plt.title(f'z_height, env {env_num}')
    plt.xlabel('Episode')
    plt.ylabel('Max height in episode')
    plot = convert_matplotlib_fig_to_array(plt.gcf())
    logger.image(f'{bufferid}/Max Height', plot[None])
    plt.close()

    if 'agent0_steer_velocity' in all_df.keys():
        plt.figure()
        plt.plot(df['episode'].to_numpy().astype(float), df['agent0_steer_velocity'].to_numpy().astype(float), '.')
        plt.title(f'Steer velocity, env {env_num}')
        plt.xlabel('Episode')
        plt.ylabel('Rotational velocity')
        plot = convert_matplotlib_fig_to_array(plt.gcf())
        logger.image(f'{bufferid}/Rotational velocity', plot[None])

        plt.close()
    ## TODO: Plot wall collisions
    print(all_df)
    ## TODO: Plot time spent in each quadrant
    logger.write()
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