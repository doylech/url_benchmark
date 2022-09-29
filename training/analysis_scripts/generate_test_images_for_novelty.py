"""
Make a set of test images with novel and familiar balls (see generate_test_images_for_novelty.py).
"""
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
import time
import pickle
import glob

import argparse
from tqdm import tqdm

from os.path import expanduser
import training.utils.loading_utils as lu
import training.utils.diagnosis_utils as du
import training.utils.img_utils as iu

tf.config.run_functions_eagerly(True)

if __name__ == "__main__":
  testbufferid = 'DRE-50'
  replay_buffer_name = 'train_replay_0'

  home = expanduser("~")
  buffer_basedir = f"{home}/logs/{testbufferid}"

  save_dir = f"{home}/logs/test_images_from_{testbufferid}/"
  os.makedirs(save_dir, exist_ok=True)

  # Load in eps
  which_eps = np.arange(0, 100, 4)
  which_t = np.arange(0, 500, 20)
  iu.save_bmp_from_eps(buffer_basedir, replay_buffer_name, save_dir, which_eps, which_t)

  # Load in images, to verify that things are working.
  load_dir = f"{home}/logs/test_images_from_{testbufferid}/"
  load_imgs, imfs = iu.load_bmp(os.path.join(load_dir, 'yellow'))

  print('done')