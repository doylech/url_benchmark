"""
Manually fiddle around with the latent space of Dreamer, in a trained model.
This is called from run_dreamer.py
# TODO: Be able to load a world model in directly from a checkpoint, outside of run_dreamer.py
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def fiddle_latent_space(wm, state, action_space):
    feat = wm.dynamics.get_feat(state[-1][0])
    plt.plot(np.arange(np.prod(feat.numpy().shape)), feat.numpy().squeeze()), \
    plt.title('Latent state'), plt.show()
    recon = wm.heads['image'](feat).mode() + 0.5
    plt.imshow(recon.numpy().squeeze()), plt.show()

    d = [5, 15, 30, 239]
    feat_mod = tf.identity(feat) - 10 * tf.reduce_sum(tf.one_hot(d, feat.shape[1]), axis=0)
    recon = wm.heads['image'](feat_mod).mode() + 0.5
    plt.imshow(recon.numpy().squeeze()), plt.title(f'Push latent along {d}'), plt.show()

    feat_rand = tf.random.uniform(feat.shape) - 0.5
    recon = wm.heads['image'](feat_rand).mode() + 0.5
    plt.imshow(recon.numpy().squeeze()), plt.title('Random latent sample'), plt.show()

    # Now trying taking an action step
    prior = state[-1][0].copy()
    for i in range(20):
        plt.figure()
        rand_action = 1 * action_space.sample()
        rand_action = tf.Variable(rand_action[np.newaxis, np.newaxis, :])
        prior = wm.dynamics.imagine(rand_action, prior)
        openl = wm.heads['image'](wm.dynamics.get_feat(prior)).mode() + 0.5
        prior = {k: tf.squeeze(v, axis=0) for k, v in prior.items()}
        plt.imshow(openl.numpy().squeeze()), plt.title(i)
    plt.show()