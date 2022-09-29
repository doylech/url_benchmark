"""
Actually decode behavioral variables from latent state, using
dataset that was organized in decode_from_latent_step1.py
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy as sp
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
import pandas as pd

if __name__ == "__main__":
  expid = 'GEN-308'
  basedir = f"/home/saal2/logs/{expid}"
  env_num = 0

  vars = np.load(f'{basedir}/vars_for_decode.npz')
  xyz = vars['xyz']
  logit = vars['logit']
  deter = vars['deter']
  which_ep = vars['which_ep'].flatten()
  nt = xyz.shape[0]

  # Make train/test sets
  np.random.seed(1)
  do_by_episode = True
  if do_by_episode:
    eps = np.random.permutation(np.unique(which_ep))
    ntrain = int(len(eps)*0.6)
    train_eps = eps[:ntrain]
    test_eps = eps[ntrain:]
    train_inds = np.where(sum([which_ep==x for x in train_eps]))[0]
    test_inds = np.where(sum([which_ep==x for x in test_eps]))[0]
  else:
    inds = np.random.permutation(np.arange(nt))
    ntrain = int(nt*0.6)
    train_inds = inds[:ntrain]
    test_inds = inds[ntrain:]

  labels = ['x', 'y']
  for ind in [0, 1]:
    Y = xyz[:, ind]
    X = deter

    # TODO: Factor this, so that can run for multiple Y

    Xtrain = X[train_inds]
    Ytrain = Y[train_inds]
    Xtest =  X[test_inds]
    Ytest =  Y[test_inds]

    # Let's just try linear regression...
    for model in [LinearRegression(), Ridge(alpha=1.0), Lasso(alpha=1.0)]:
      clf = model.fit(Xtrain, Ytrain)
      score_train = clf.score(Xtrain, Ytrain)
      score_test =  clf.score(Xtest, Ytest)
      ypred_test = clf.predict(Xtest)

      print(model)
      print(f'Train: {score_train}, Test: {score_test}')
      plt.plot(Ytest, ypred_test, '.')
      plt.xlabel(f'True {labels[ind]}')
      plt.ylabel(f'Pred {labels[ind]}')
      if do_by_episode:
        plt.title(f'{expid}, {model}, {score_test:.3f}, episode_shuffle')
      else:
        plt.title(f'{expid}, {model}, {score_test:.3f}, step_shuffle')

      plt.xlim([-20, 20])
      plt.ylim([-20, 20])

      plt.show()

  print('done')