"""Saveout of a gif of a specified episode from the replay buffer."""
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import ImageSequenceClip

if __name__ == "__main__":
  basedir =  '/home/saal2/logs/p2e_fiddle_dreamer-launch'
  path = f'{basedir}/logs/GEN-193/train_replay_0/20210919T234502-300c15c12b054ca2ba5d55b4e4e1a090-356.npz'
  path = f'{basedir}/20210920T092116-2577daef5f074941a75a940ef35a558b-388.npz'
  path = f'{basedir}/20210921T234431-d27fbaaeb8224e89807c6a6e16513a8f-78.npz'
  path = f'{basedir}/20210922T010106-37122022774342abb0b6798ac24ce794-293.npz'
  path = f'{basedir}/logs/GEN-EXAMPLE_EPS/train_replay/20211222T185313-ee59ea03cd2e4d2cbc6f24e1ef5f6399-1000.npz'
  path = f'{basedir}/train_replay/20211222T213701-cd1d568ab7f340b0963a04a7821954b5-1000.npz'
  path = f'{basedir}/train_replay_0/20220318T161416-22735f6703074d56a059aa6d98fc447f-25.npz'
  path = f'{basedir}/train_replay_0/20220819T075218-4997860c482a4548a890383f51aaa58c-99'

  ep = np.load(f'{path}.npz')
  imgs = ep['image']

  clip = ImageSequenceClip(list(imgs), fps=20)
  # clip.write_gif(f'{basedir}/ep.gif', fps=20)
  clip.write_gif(f'{path}.gif', fps=20)

  n=0
  plt.figure()
  plt.subplot(2,1,1)
  plt.plot(np.log10(np.abs(np.sum(ep['agent0/sensors_velocimeter'][n:,:], axis=1))), '-o'), plt.title('log10(abs(sum(vel))')

  plt.subplot(2,1,2)
  plt.plot(np.log10(np.abs(np.sum(ep['agent0/sensors_accelerometer'][n:,:], axis=1))), '-o'),  plt.title('log10(abs(sum(accel))')

  n=-10
  plt.figure()
  plt.subplot(2,1,1)
  plt.plot(np.log10(np.abs(np.sum(ep['agent0/sensors_velocimeter'][n:,:], axis=1))), '-o'), plt.title('log10(abs(sum(vel))')

  plt.subplot(2,1,2)
  plt.plot(np.log10(np.abs(np.sum(ep['agent0/sensors_accelerometer'][n:,:], axis=1))), '-o'),  plt.title('log10(abs(sum(accel))')

  plt.show()

  n=0
  plt.figure()
  plt.subplot(2,1,1)
  plt.plot(np.log10(np.abs(np.max(ep['agent0/sensors_velocimeter'][n:,:], axis=1))), '-o'), plt.title('log10(abs(max(vel))')

  plt.subplot(2,1,2)
  plt.plot(np.log10(np.abs(np.max(ep['agent0/sensors_accelerometer'][n:,:], axis=1))), '-o'), plt.title('log10(abs(max(accel))')


  n=-10
  plt.figure()
  plt.subplot(2,1,1)
  plt.plot(np.log10(np.abs(np.max(ep['agent0/sensors_velocimeter'][n:,:], axis=1))), '-o'), plt.title('log10(abs(max(vel))')

  plt.subplot(2,1,2)
  plt.plot(np.log10(np.abs(np.max(ep['agent0/sensors_accelerometer'][n:,:], axis=1))), '-o'), plt.title('log10(abs(max(accel))')

  plt.show()

  print('done')