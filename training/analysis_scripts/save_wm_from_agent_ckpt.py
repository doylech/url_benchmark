"""
Load an agent checkpoint, and save a separate wm checkpoint.
"""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
import time
import pickle

import elements
import common
from dreamerv2 import agent
import argparse
from tqdm import tqdm

from os.path import expanduser
import training.utils.loading_utils as lu
import training.utils.diagnosis_utils as du

def save_wm(expid, checkpoint_name):
  home = expanduser("~")
  basedir = f"{home}/logs/{expid}"

  from tensorflow.keras.mixed_precision import experimental as prec
  prec.set_policy(prec.Policy('mixed_float16'))

  ## Load agent from checkpoint
  agnt = lu.load_agent(basedir,
                       checkpoint_name=checkpoint_name,
                       batch_size=5,
                       deterministic=False
                       )

  # Save world model.
  agnt.wm.save(os.path.join(basedir, checkpoint_name[:-4] + '_wm.pkl'))
  agnt._task_behavior.save(os.path.join(basedir, checkpoint_name[:-4] + '_taskb.pkl'))
  agnt._expl_behavior.save(os.path.join(basedir, checkpoint_name[:-4] + '_explb.pkl'))

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Specify args.')
  parser.add_argument('--expid', dest='expid', default='GEN-538'
                      )
  parser.add_argument('--ckpt', dest='checkpoint_name',
                      default='variables_train_agent_envindex0_010002500.pkl'
                      )
  args = parser.parse_args()

  expid = args.expid
  checkpoint_name = args.checkpoint_name
  save_wm(expid, checkpoint_name)

  print('done')