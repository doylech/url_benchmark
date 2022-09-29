import pickle
import numpy as np
import os
from os.path import expanduser

if __name__ == "__main__":
  expid = 'GEN-801'

  home = expanduser("~")
  basedir = f"{home}/logs/{expid}"

  with open(os.path.join(basedir, 'wm_vars.pkl'), 'rb') as f:
    vars_local = pickle.load(f)

  with open(os.path.join(basedir, 'wm_vars_remote.pkl'), 'rb') as f:
    vars_remote = pickle.load(f)


  for key in vars_local.keys():
    print(key)
    diff = np.max(np.abs(vars_local[key] - vars_remote[key]))
    assert(diff == 0)
    print(diff)
  print('done')