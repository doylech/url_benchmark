import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def set_interactive_plot(interactive):
  import matplotlib as mpl
  mpl.use('TkAgg') if interactive else mpl.use('module://backend_interagg')

def simple_plot(ax):
  ax.spines.right.set_visible(False)
  ax.spines.top.set_visible(False)
  ax.yaxis.set_ticks_position('left')
  ax.xaxis.set_ticks_position('bottom')


node_mat = []
for node, inds in node_visits.items():
  visits = np.zeros(nt)
  visits[inds] = 1
  node_mat.append(visits)
node_mat = np.array(node_mat)
node_id_mat = node_mat * np.arange(1, node_mat.shape[0] + 1)[:, np.newaxis]
node_vec = np.sum(node_id_mat, axis=0)


def organize_visits(which_visits, nt):
  """Organize roi visits into a vector, where entry represents which roi."""
  mat = []
  for node, inds in which_visits.items():
    visits = np.zeros(nt)
    visits[inds] = 1
    mat.append(visits)
  mat = np.vstack(mat)
  id_mat = mat * np.arange(1, mat.shape[0] + 1)[:, np.newaxis]
  vec = np.sum(id_mat, axis=0)
  return vec

def get_turn_bias(stem_visits, intersection_visits, right_visits, left_visits,
                  nt, saverate, do_plot=True):

  tt = saverate * np.arange(nt)
  stem_vec = organize_visits(stem_visits, nt)
  intersection_vec = organize_visits(intersection_visits, nt)
  left_vec = organize_visits(left_visits, nt)
  right_vec = organize_visits(right_visits, nt)

  STEM = 0
  LEFT = 1
  RIGHT = 2
  INTX = 3
  vec = np.vstack([stem_vec, left_vec, right_vec, intersection_vec])
  vec = vec[:, np.where(np.sum(vec, axis=0) > 0)[0]] # Get rid of all timepoints where not in any roi.
  seq = []
  last_v = vec[:, 0]
  for k in range(vec.shape[1]):
    v = vec[:, k]
    if np.all(v == 0):
      continue
    if np.all(v == last_v):
      continue
    seq.append(v)
    last_v = v
  seq = np.vstack(seq).T

  # Probability of moving forward from a stem instead of reversing
  stem_entries = np.where(np.logical_and(seq[STEM, :-1] > 0, seq[INTX, 1:] > 0))[0]
  stem_returns = np.where(np.logical_and(np.logical_and(seq[STEM, :-2] > 0, seq[INTX, 1:-1] > 0), seq[STEM, 2:] > 0))[0]
  Psf = 1 - (len(stem_returns) / len(stem_entries))

  # Probability of an alternating turn from the preceding one
  left_turns = np.logical_and(np.logical_and(seq[STEM, :-2] > 0, seq[INTX, 1:-1] > 0), seq[LEFT, 2:] > 0).astype(int)
  right_turns = np.logical_and(np.logical_and(seq[STEM, :-2] > 0, seq[INTX, 1:-1] > 0), seq[RIGHT, 2:] > 0).astype(int)
  turns = 1 * left_turns + 2 * right_turns
  turns = turns[np.where(turns > 0)[0]]
  alternating_turns = np.where(np.abs(np.diff(turns)) > 0)[0]
  non_alternating_turns = np.where(np.abs(np.diff(turns)) == 0)[0]
  Psa = len(alternating_turns) / (len(non_alternating_turns) + len(alternating_turns))

  # Probability of moving forward from a branch instead of reversing
  l_to_r = np.logical_and(np.logical_and(seq[LEFT, :-2] > 0, seq[INTX, 1:-1] > 0), seq[RIGHT, 2:] > 0).astype(int)
  l_to_s = np.logical_and(np.logical_and(seq[LEFT, :-2] > 0, seq[INTX, 1:-1] > 0), seq[STEM, 2:] > 0).astype(int)
  l_to_l = np.logical_and(np.logical_and(seq[LEFT, :-2] > 0, seq[INTX, 1:-1] > 0), seq[LEFT, 2:] > 0).astype(int)
  r_to_l = np.logical_and(np.logical_and(seq[RIGHT, :-2] > 0, seq[INTX, 1:-1] > 0), seq[LEFT, 2:] > 0).astype(int)
  r_to_s = np.logical_and(np.logical_and(seq[RIGHT, :-2] > 0, seq[INTX, 1:-1] > 0), seq[STEM, 2:] > 0).astype(int)
  r_to_r = np.logical_and(np.logical_and(seq[RIGHT, :-2] > 0, seq[INTX, 1:-1] > 0), seq[RIGHT, 2:] > 0).astype(int)
  nbranch_forward = len(np.where(l_to_r)[0]) + len(np.where(r_to_l)[0]) + \
                    len(np.where(l_to_s)[0]) + len(np.where(r_to_s)[0])
  nbranch_reverse = len(np.where(l_to_l > 0)[0]) + len(np.where(r_to_r > 0)[0])
  Pbf = nbranch_forward / (nbranch_forward + nbranch_reverse)

  # Probability of moving from branch to stem
  l_to_s = np.logical_and(np.logical_and(seq[LEFT, :-2] > 0, seq[INTX, 1:-1] > 0), seq[STEM, 2:] > 0).astype(int)
  r_to_s = np.logical_and(np.logical_and(seq[RIGHT, :-2] > 0, seq[INTX, 1:-1] > 0), seq[STEM, 2:] > 0).astype(int)
  l_to_r = np.logical_and(np.logical_and(seq[LEFT, :-2] > 0, seq[INTX, 1:-1] > 0), seq[RIGHT, 2:] > 0).astype(int)
  r_to_l = np.logical_and(np.logical_and(seq[RIGHT, :-2] > 0, seq[INTX, 1:-1] > 0), seq[LEFT, 2:] > 0).astype(int)
  nto_stem = len(np.where(l_to_s)[0]) + len(np.where(r_to_s)[0])
  nto_branch = len(np.where(l_to_r)[0]) + len(np.where(r_to_l)[0])
  Pbs = nto_stem / (nto_stem + nto_branch)

  print(f'Psf: {Psf:0.2f}, Psa: {Psa:0.2f}, Pbf: {Pbf:0.2f}, Pbs: {Pbs:0.2f}')

  if do_plot:
    plt.plot(Psf, Pbf, 'g+'), plt.plot(2 / 3, 2 / 3, 'b+')
    plt.xlim([0, 1]), plt.ylim([0, 1])
    plt.xlabel('Psf'), plt.ylabel('Pbf')
    plt.show()

    plt.plot(Psa, Pbs, 'g+'), plt.plot(1 / 2, 1 / 2, 'b+')
    plt.xlim([0, 1]), plt.ylim([0, 1])
    plt.xlabel('Psa'), plt.ylabel('Pbs')
    plt.show()

  return (Psf, Pbf, Psa, Pbs)

def pointInRect(xs, ys, rect):
  x1, y1, w, h = rect
  x2, y2 = x1 + w, y1 + h
  # in_rect = np.logical_and(np.logical_and(xs > x1, xs < x2), np.logical_and(ys > y1, ys < y2))
  in_rect = np.all((xs > x1, xs < x2, ys > y1, ys < y2), axis=0)
  return in_rect

def roi_visit(coord, x, y, w, h, color, do_plot, do_plot_rect, do_plot_dots=True):
  r = (coord[0] - w / 2, coord[1] - h / 2, w, h)
  inds = np.where(pointInRect(x.to_numpy(), y.to_numpy(), r))[0]
  if do_plot:
    if do_plot_dots:
      plt.plot(x[inds], y[inds], '.', color=color, markersize=1, alpha=1, linewidth=0)
    if do_plot_rect:
      rect = Rectangle((coord[0] - w / 2, coord[1] - h / 2), w, h, fill=False, edgecolor=color)
      plt.gca().add_patch(rect)
  return inds

def get_roi_visits(x, y, do_plot=True, do_plot_rect=True, title_str=''):
  """Get timestamps of each visit to various regions of interest."""
  (stem_coords, left_coords,
   right_coords, intersections1_coords,
   endnode_coords, nonmaze_coords) = get_labyrinth_coords(stem_d=0.9)

  # Set up regions of interest
  w_stem = 0.8
  h_stem = 0.8
  w_intersections = 0.9
  h_intersections = 0.9
  w_end = 3
  h_end = 1.5
  w_nonmaze = 12.5
  h_nonmaze = 26.5

  if do_plot:
    plt.plot(x, y, 'k.', markersize=1, alpha=0.2)
    ax = plt.gca()

  node_visits = {}
  intersection_visits = {}
  homecage_visits = {}
  stem_visits = {}
  right_visits = {}
  left_visits = {}
  for i, coord in enumerate(endnode_coords):
    node_visits[i] = roi_visit(coord, x, y, w_end, h_end, 'r', do_plot, do_plot_rect)
  for i, coord in enumerate(intersections1_coords):
    intersection_visits[i] = roi_visit(coord, x, y, w_intersections, h_intersections, 'g', do_plot, do_plot_rect)
  for i, coord in enumerate(stem_coords):
    stem_visits[i] = roi_visit(coord, x, y, w_stem, h_stem, 'm', do_plot, do_plot_rect)
  for i, coord in enumerate(right_coords):
    right_visits[i] = roi_visit(coord, x, y, w_stem, h_stem, 'c', do_plot, do_plot_rect)
  for i, coord in enumerate(left_coords):
    left_visits[i] = roi_visit(coord, x, y, w_stem, h_stem, 'orange', do_plot, do_plot_rect)
  for i, coord in enumerate(nonmaze_coords):
    coord = np.array(coord) + np.array([w_nonmaze/2, h_nonmaze/2])
    homecage_visits[i] = roi_visit(coord, x, y, w_nonmaze, h_nonmaze, 'b', do_plot, do_plot_rect, do_plot_dots=False)

  if do_plot:
    nt = len(x)
    plt.title(title_str)
    plt.show()

  return (node_visits, intersection_visits,
          homecage_visits, stem_visits,
          right_visits, left_visits)


def get_time_in_maze(homecage_visits, nt, bin_seconds, CONTROL_TIMESTEP, do_plot=True, title_str=''):
  """Compute time spent in the maze (as opposed to homecage)."""
  homecage_occupancy = np.zeros(nt)
  homecage_occupancy[homecage_visits] = 1
  maze_occupancy = 1 - homecage_occupancy

  binsize = int(bin_seconds / CONTROL_TIMESTEP)
  nbins = int(np.floor(len(maze_occupancy) / binsize))
  binned_maze_occupancy = np.zeros(nbins)
  for i in np.arange(nbins):
    binned_maze_occupancy[i] = np.sum(maze_occupancy[binsize * i:binsize * (i + 1)]) / binsize

  if do_plot:
    plt.plot(np.arange(nbins) * bin_seconds, binned_maze_occupancy)
    plt.ylim([0, 1])
    plt.xlim([0, nbins * bin_seconds])
    plt.ylabel('% time spent in maze')
    plt.xlabel('Absolute time (s)')
    simple_plot(plt.gca())
    plt.title(title_str)
    plt.tight_layout()
    plt.show()

  return binned_maze_occupancy

def get_labyrinth_coords(stem_d=0.9):
  y_stems1 = np.array([19.6, 8.4, -2.8, -14]) - stem_d
  x_stems1 = np.array([-9.775, 1.35, 12.55, 23.775])
  y_left1 = np.array([19.6, 8.4, -2.8, -14])
  x_left1 = np.array([-9.775, 1.35, 12.55, 23.775]) - stem_d
  y_right1 = np.array([19.6, 8.4, -2.8, -14])
  x_right1 = np.array([-9.775, 1.35, 12.55, 23.775]) + stem_d
  stem_coords = list(itertools.product(x_stems1, y_stems1))
  left_coords = list(itertools.product(x_left1, y_left1))
  right_coords = list(itertools.product(x_right1, y_right1))

  y_stems1 = np.array([14, 2.8, -8.4, -19.6]) + stem_d
  x_stems1 = np.array([-9.775, 1.35, 12.55, 23.775])
  y_left1 = np.array([14, 2.8, -8.4, -19.6])
  x_left1 = np.array([-9.775, 1.35, 12.55, 23.775]) + stem_d
  y_right1 = np.array([14, 2.8, -8.4, -19.6])
  x_right1 = np.array([-9.775, 1.35, 12.55, 23.775]) - stem_d
  stem_coords.extend(list(itertools.product(x_stems1, y_stems1)))
  left_coords.extend(list(itertools.product(x_left1, y_left1)))
  right_coords.extend(list(itertools.product(x_right1, y_right1)))

  y_stems1 = np.array([-16.8, -5.6, 5.6, 16.8])
  x_stems1 = np.array([-9.775, 12.55]) + stem_d
  y_left1 = np.array([-16.8, -5.6, 5.6, 16.8]) - stem_d
  x_left1 = np.array([-9.775, 12.55])
  y_right1 = np.array([-16.8, -5.6, 5.6, 16.8]) + stem_d
  x_right1 = np.array([-9.775, 12.55])
  stem_coords.extend(list(itertools.product(x_stems1, y_stems1)))
  left_coords.extend(list(itertools.product(x_left1, y_left1)))
  right_coords.extend(list(itertools.product(x_right1, y_right1)))

  y_stems1 = np.array([-16.8, -5.6, 5.6, 16.8])
  x_stems1 = np.array([1.35, 23.775]) - stem_d
  y_left1 = np.array([-16.8, -5.6, 5.6, 16.8]) + stem_d
  x_left1 = np.array([1.35, 23.775])
  y_right1 = np.array([-16.8, -5.6, 5.6, 16.8]) - stem_d
  x_right1 = np.array([1.35, 23.775])
  stem_coords.extend(list(itertools.product(x_stems1, y_stems1)))
  left_coords.extend(list(itertools.product(x_left1, y_left1)))
  right_coords.extend(list(itertools.product(x_right1, y_right1)))

  y_stems1 = np.array([16.8, -5.6]) - stem_d
  x_stems1 = np.array([-4.2125, 18.1625])
  y_left1 = np.array([16.8, -5.6])
  x_left1 = np.array([-4.2125, 18.1625]) - stem_d
  y_right1 = np.array([16.8, -5.6])
  x_right1 = np.array([-4.2125, 18.1625]) + stem_d
  stem_coords.extend(list(itertools.product(x_stems1, y_stems1)))
  left_coords.extend(list(itertools.product(x_left1, y_left1)))
  right_coords.extend(list(itertools.product(x_right1, y_right1)))

  y_stems1 = np.array([-16.8, 5.6]) + stem_d
  x_stems1 = np.array([-4.2125, 18.1625])
  y_left1 = np.array([-16.8, 5.6])
  x_left1 = np.array([-4.2125, 18.1625]) + stem_d
  y_right1 = np.array([-16.8, 5.6])
  x_right1 = np.array([-4.2125, 18.1625]) - stem_d
  stem_coords.extend(list(itertools.product(x_stems1, y_stems1)))
  left_coords.extend(list(itertools.product(x_left1, y_left1)))
  right_coords.extend(list(itertools.product(x_right1, y_right1)))

  y_stems1 = np.array([-11.2, 11.2])
  x_stems1 = np.array([-4.2125]) + stem_d
  y_left1 = np.array([-11.2, 11.2]) - stem_d
  x_left1 = np.array([-4.2125])
  y_right1 = np.array([-11.2, 11.2]) + stem_d
  x_right1 = np.array([-4.2125])
  stem_coords.extend(list(itertools.product(x_stems1, y_stems1)))
  left_coords.extend(list(itertools.product(x_left1, y_left1)))
  right_coords.extend(list(itertools.product(x_right1, y_right1)))

  y_stems1 = np.array([-11.2, 11.2])
  x_stems1 = np.array([18.1625]) - stem_d
  y_left1 = np.array([-11.2, 11.2]) + stem_d
  x_left1 = np.array([18.1625])
  y_right1 = np.array([-11.2, 11.2]) - stem_d
  x_right1 = np.array([18.1625])
  stem_coords.extend(list(itertools.product(x_stems1, y_stems1)))
  left_coords.extend(list(itertools.product(x_left1, y_left1)))
  right_coords.extend(list(itertools.product(x_right1, y_right1)))

  y_stems1 = np.array([11.2]) - stem_d
  x_stems1 = np.array([6.95])
  y_left1 = np.array([11.2])
  x_left1 = np.array([6.95]) - stem_d
  y_right1 = np.array([11.2])
  x_right1 = np.array([6.95]) + stem_d
  stem_coords.extend(list(itertools.product(x_stems1, y_stems1)))
  left_coords.extend(list(itertools.product(x_left1, y_left1)))
  right_coords.extend(list(itertools.product(x_right1, y_right1)))

  y_stems1 = np.array([-11.2]) + stem_d
  x_stems1 = np.array([6.95])
  y_left1 = np.array([-11.2])
  x_left1 = np.array([6.95]) + stem_d
  y_right1 = np.array([-11.2])
  x_right1 = np.array([6.95]) - stem_d
  stem_coords.extend(list(itertools.product(x_stems1, y_stems1)))
  left_coords.extend(list(itertools.product(x_left1, y_left1)))
  right_coords.extend(list(itertools.product(x_right1, y_right1)))

  y_stems1 = np.array([0])
  x_stems1 = np.array([6.95]) - stem_d
  y_left1 = np.array([0]) + stem_d
  x_left1 = np.array([6.95])
  y_right1 = np.array([0]) - stem_d
  x_right1 = np.array([6.95])
  stem_coords.extend(list(itertools.product(x_stems1, y_stems1)))
  left_coords.extend(list(itertools.product(x_left1, y_left1)))
  right_coords.extend(list(itertools.product(x_right1, y_right1)))

  y_intersections1 = [-19.6, -14, -8.4, -2.8, 2.8, 8.4, 14, 19.6]
  x_intersections1 = [-9.775, 1.35, 12.55, 23.775]
  intersections1_coords = list(itertools.product(x_intersections1, y_intersections1))
  y_intersections2 = [-16.8, -5.6, 5.6, 16.8]
  x_intersections2 = [-9.775, -4.2125, 1.35, 12.55, 18.1625, 23.775]
  intersections2_coords = list(itertools.product(x_intersections2, y_intersections2))
  y_intersections3 = [-11.2, 11.2]
  x_intersections3 = [-4.2125, 6.95, 18.1625]
  intersections3_coords = list(itertools.product(x_intersections3, y_intersections3))
  y_intersections4 = [0]
  x_intersections4 = [6.95]
  intersections4_coords = list(itertools.product(x_intersections4, y_intersections4))
  intersections1_coords.extend(intersections2_coords)
  intersections1_coords.extend(intersections3_coords)
  intersections1_coords.extend(intersections4_coords)

  y_endnodes = [-19.6, -14, -8.4, -2.8, 2.8, 8.4, 14, 19.6]
  x_endnodes = [-12.75, -6.8, -1.6, 4.3, 9.6, 15.5, 20.8, 26.75]
  endnode_coords = list(itertools.product(x_endnodes, y_endnodes))

  x_nonmaze = -27.25
  y_nonmaze = -13.25
  nonmaze_coords = [(x_nonmaze, y_nonmaze)]

  return (stem_coords, left_coords, right_coords,
          intersections1_coords, endnode_coords, nonmaze_coords)