"""Using data from https://github.com/Herseninstituut/Ahmadlou_etal_Science_2021
Using datafiles processed by analyze_mouse_novel_objects.m"""
import os
import glob
import scipy.io as sio
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

if __name__ == "__main__":
  data_path = '/home/saal2/logs/ahmadlou_2021'
  mouse_type = 'B6'
  mouse_type = 'GAD2_ZI_Exc'
  mouse_type = 'GAD2_ZI_Inh'
  mouse_type = 'GAD2_ZI_Cont'
  mouse_type = 'GAD2_ZI_Cont_10'
  mouse_type = 'PLZI_Cont_CNO'
  mouse_type = 'BL6_PLZI_CNO'
  files = glob.glob(os.path.join(data_path, f'{mouse_type}*NovOld*.mat'))

  print(files)

  # Make a dict
  data = defaultdict(dict)
  for f in files:
    fname = f.split('/')[-1]
    mouse_id = fname.split('-')[0] + '_' + fname.split('-')[1]
    expt_num = fname.split('-')[-1].split('.')[0]

    print(f'{mouse_id}, {expt_num}')
    d = sio.loadmat(f)
    data[mouse_id][expt_num] = d


  # Now compile visitation matrix across mice
  T = 660;
  s = 100;
  categories = ['Approach', 'Sniff', 'Bite', 'Grab', 'Carry']
  # categories = ['Sniff', 'Bite', 'Grab', 'Carry']
  flat_data = defaultdict(dict)
  for m in data.keys():
    for expt in data[m].keys():
      d = data[m][expt]

      ints = {}
      for obj in ['NEW', 'OLD']:
        ints[obj] = np.zeros(T*s)

        for c in categories:
          start = (d[f'{c}{obj}_start']*s).astype(int)
          end = (d[f'{c}{obj}_end']*s).astype(int)
          if len(start) > 0:
            start = start[0]
            end = end[0]
            for ii in range(len(start)):
              ints[obj][start[ii]:end[ii]] = 1
      flat_data[m][int(expt)] = {'new': ints['NEW'], 'old': ints['OLD']}


  for combine_all in [1]:
    if combine_all:
      # Now plot all objects (not just first object)
      all_expt_new = []
      all_expt_old = []
      for i, m in enumerate(flat_data.keys()):
        for which_expt in flat_data[m].keys():
          all_expt_new.append(flat_data[m][which_expt]['new'])
          all_expt_old.append(flat_data[m][which_expt]['old'])
      all_expt_new = np.vstack(all_expt_new)
      all_expt_old = np.vstack(all_expt_old)
      which_expt = 'all'
    else:
      which_expt = 1
      all_expt_new = []
      all_expt_old = []
      for i, m in enumerate(flat_data.keys()):
        try:
          all_expt_new.append(flat_data[m][which_expt]['new'])
          all_expt_old.append(flat_data[m][which_expt]['old'])
        except:
          pass
      all_expt_new = np.vstack(all_expt_new)
      all_expt_old = np.vstack(all_expt_old)

    tt = np.arange(all_expt_new.shape[1])/s
    plt.figure(figsize=(7, 5))
    plt.subplot(2,1,1)
    plt.imshow(all_expt_new, aspect='auto', interpolation='nearest', extent=[tt[0], tt[-1], 0, all_expt_new.shape[0]])
    plt.title('Novel object')
    plt.ylabel('Mouse #')
    plt.xlim([0, 600])
    plt.subplot(2,1,2)
    plt.imshow(all_expt_old, aspect='auto', interpolation='nearest', extent=[tt[0], tt[-1], 0, all_expt_new.shape[0]])
    plt.title('Old object')
    plt.ylabel('Mouse #')
    plt.xlabel('Time (s)')
    plt.xlim([0, 600])
    plt.suptitle(f'{mouse_type}: Object {which_expt}, n={all_expt_new.shape[0]}')
    plt.tight_layout()
    plt.show()

    plt.figure()
    legend_p = []
    m_new = np.mean(np.cumsum(all_expt_new, axis=1), axis=0)
    s_new = scipy.stats.sem(np.cumsum(all_expt_new, axis=1), axis=0)
    m_old = np.mean(np.cumsum(all_expt_old, axis=1), axis=0)
    s_old = scipy.stats.sem(np.cumsum(all_expt_old, axis=1), axis=0)
    plt.fill_between(tt, m_new-s_new, m_new+s_new, alpha=0.5, color='r')
    plt.fill_between(tt, m_old-s_old, m_old+s_old, alpha=0.5, color='b')
    p1, = plt.plot(tt, m_new, color='r')
    p2, = plt.plot(tt, m_old, color='b')
    plt.legend([p1, p2], ['New', 'Old'], loc='upper left')
    plt.ylabel('Interactions')
    plt.title(f'Interaction categories: {categories}')
    plt.suptitle(f'{mouse_type}: Object {which_expt}, n={all_expt_new.shape[0]}')
    plt.xlabel('Time(s)')
    plt.ylim([0, 8000])
    plt.show()

  print('done')