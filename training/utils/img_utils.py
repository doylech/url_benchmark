"""Utilities used in generate_test_images_for_novelty.py"""
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import training.utils.loading_utils as lu

def has_color(image, color):
  """Assess whether an image has a specified color ("""
  if color == 'yellow':
    color_max = [255, 255, 20]
    color_min = [160, 160, 0]
  elif color == 'magenta':
    color_max = [255, 20, 255]
    color_min = [75, 0, 75]
  else:
    raise(NotImplementedError)

  color = np.where((image[:, :, 0] >= color_min[0]) &
                   (image[:, :, 1] >= color_min[1]) &
                   (image[:, :, 2] >= color_min[2]) &
                   (image[:, :, 0] <= color_max[0]) &
                   (image[:, :, 1] <= color_max[1]) &
                   (image[:, :, 2] <= color_max[2])
                   )[0]
  return len(color) > 0

def load_bmp(load_dir):
  """Load a folder full of .bmp images."""
  imfs = glob.glob(os.path.join(load_dir, '*.bmp'))
  imfs.sort()
  im0 = plt.imread(imfs[0])
  load_imgs = np.zeros((len(imfs), im0.shape[0], im0.shape[1], im0.shape[2]), dtype=np.uint8)
  for i, imf in enumerate(imfs):
    load_imgs[i, :, :, :] = plt.imread(imf)

  return load_imgs, imfs

def save_bmp_from_eps(buffer_basedir, replay_buffer_name, save_dir, which_eps, which_t):
  """Save images from episodes, categorized into different folders
  based on the colors present in each image."""
  os.makedirs(os.path.join(save_dir, 'yellow'), exist_ok=True)
  os.makedirs(os.path.join(save_dir, 'magenta'), exist_ok=True)
  os.makedirs(os.path.join(save_dir, 'noball'), exist_ok=True)
  os.makedirs(os.path.join(save_dir, 'yellow_magenta'), exist_ok=True)

  for batch_ep in which_eps:
    eps = lu.load_eps(buffer_basedir, replay_buffer_name, batch_eps=[batch_ep])
    imgs = eps[0]['image']
    for i in which_t:
      img = imgs[i, :, :, :]
      has_yellow = has_color(img, 'yellow')
      has_magenta = has_color(img, 'magenta')
      if has_yellow and has_magenta:
        plt.imsave(os.path.join(save_dir, 'yellow_magenta', f'{batch_ep:03}_{i:03}.bmp'), img)
      elif has_yellow:
        plt.imsave(os.path.join(save_dir, 'yellow', f'{batch_ep:03}_{i:03}.bmp'), img)
        print(i)
      elif has_magenta:
        plt.imsave(os.path.join(save_dir, 'magenta', f'{batch_ep:03}_{i:03}.bmp'), img)
      else:
        plt.imsave(os.path.join(save_dir, 'noball', f'{batch_ep:03}_{i:03}.bmp'), img)