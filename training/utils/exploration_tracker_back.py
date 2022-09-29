"""
HOOK_NAMES = ('initialize_episode_mjcf',
              'after_compile',
              'initialize_episode',
              'before_step',
              'before_substep',
              'after_substep',
              'after_step')
"""
import math

from bidict import bidict
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from PIL import Image

from scipy import sparse

import glob
import os

class ExplorationTracker:
    def __init__(self,
                 env_name: str,
                 run_id: int,
                 walkers,
                 primary_agent,
                 params: dict
                 ):

        self.env_name = env_name
        self.walkers = walkers
        self.primary_agent = primary_agent
        self.run_id = run_id

        self.total_step = 0
        self.episode_step = 0
        self.first_csv_write = True

        self.geom_id = bidict()
        self.collision_tracker = None
        self.episode_data = None

        self.plots = {}
        self.image_directory = 'debug/images-' + str(run_id) + '/'

        self.logger = params['logger']

        self.grid_density = params['grid_density']
        self.episode_length = params['episode_length']

        self.env_raw_output_file_name = params['env_raw_output_file_name']
        self.record_every_k_timesteps = params['record_every_k_timesteps']
        self.episodes_for_summary_metrics = params['episodes_for_summary_metrics']

        #self._clear_image_directory()
        #self._create_image_directory()

        if env_name == 'sphero-playground-TEST':

            x_size = 100
            y_size = 100

            self.shape = (x_size+1, y_size+1)

            self.plots['raw_state_count'] = np.zeros(self.shape, dtype=np.int)
            self.plots['raw_episode_state_count'] = np.zeros(self.shape, dtype=np.int)
            self.plots['intrinsic_reward'] = np.zeros(self.shape, dtype=np.float)

            self._raw_to_pixel = lambda s: (int(s[0] + x_size/2),
                                           int(s[1] + y_size/2))

            all_pos = []
            outer_bounds = []

            MAX_EPISODE_STEPS = 10

            for x in range(x_size):
                for y in range(y_size):
                    if abs(x - x_size/2) + abs(y - y_size/2) <= MAX_EPISODE_STEPS:
                        all_pos.append([x - x_size/2, y - y_size/2])


                    if abs(x - x_size/2) + abs(y - y_size/2) == x_size/2:
                        outer_bounds.append([x - x_size/2, y - y_size/2])

            self.all_pos = np.array(all_pos, dtype=np.int)
            self.outer_bounds = np.array(outer_bounds, dtype=np.int)

    def after_step(self, physics, random_state):
        self.episode_step += 1
        self.total_step += 1

        for contact in physics.data.contact:
            if contact['geom1'] == self.geom_id['agent0/shell']:
                contact_with_primary_agent = True
                other_geom_id = contact['geom2']
            elif contact['geom2'] == self.geom_id['agent0/shell']:
                contact_with_primary_agent = True
                other_geom_id = contact['geom1']
            else:
                contact_with_primary_agent = False
                other_geom_id = None

            if contact_with_primary_agent:
                # Is what primary agent collided with in set of 'important' geom_ids?
                if other_geom_id in self.geom_id.values():
                    self.collision_tracker[self.geom_id.inverse[other_geom_id]] += 1

        if self.episode_step % self.record_every_k_timesteps == 0:

            row = {}

            for walker in self.walkers:
                bound_walker = physics.bind(self.walkers[walker].root_body)

                row[f'agent{walker}_xloc'] = bound_walker.xpos[0]
                row[f'agent{walker}_yloc'] = bound_walker.xpos[1]
                row[f'agent{walker}_zloc'] = bound_walker.xpos[2]

                row[f'agent{walker}_xvel'] = bound_walker.cvel[3]
                row[f'agent{walker}_yvel'] = bound_walker.cvel[4]
                row[f'agent{walker}_zvel'] = bound_walker.cvel[5]

                if walker == self.primary_agent:
                    # To capture others, it would be [f'ball{walker}/egocentric']
                    cam_xmat = physics.named.data.cam_xmat[f'agent{walker}/egocentric']
                    cam_xmat = np.array(cam_xmat).reshape((3, 3))
                    r = Rotation.from_matrix(cam_xmat)
                    orientation = r.as_euler("ZYX")[0]
                    row[f'agent{walker}_orientation'] = orientation

            target_xpos = physics.named.data.xpos['target/']
            row[f'target_xloc'] = target_xpos[0]
            row[f'target_yloc'] = target_xpos[1]
            row[f'target_zloc'] = target_xpos[2]

            """
            bound_walker = physics.bind(self.walkers[self.primary_agent].root_body)
            walker_xloc = bound_walker.xpos[0]
            walker_yloc = bound_walker.xpos[1]
            walker_zloc = bound_walker.xpos[2]

            #convert
            cam_xmat = physics.named.data.cam_xmat['agent0/egocentric']
            cam_xmat = np.array(cam_xmat).reshape((3, 3))
            r = Rotation.from_matrix(cam_xmat)

            print(
                f'[{self.episode_step}@{self.run_id}],\t \n{row}')
                #f'pos, {round(walker_xloc, 4)}, {round(walker_yloc, 4)}, {round(walker_zloc, 4)},\t '
                #f'vel, {round(bound_walker.cvel[3], 4)}, {round(bound_walker.cvel[4], 4)}, {round(bound_walker.cvel[5], 4)}\t ')
                #f'dir {r.as_euler("ZYX")[0].round(3)}; \t'
                #f'col {self.collision_tracker.values[0]}')
            """

            self.episode_data.append(row)


    def initialize_episode(self, physics, random_state):

        print('initialize_episode hook executing in exploration tracker')

        if self.episode_data:
            df = pd.DataFrame(self.episode_data, columns=self.episode_data[0].keys())
            print('writing previous episode data')
            df.to_csv('test_out.csv', mode='a', header=self.first_csv_write)
            self.first_csv_write = False

        self.episode_step = 0

        #env.physics.model.id2name(i, 'geom')

        self.geom_id['agent0/shell'] = physics.named.model.geom_bodyid.axes.row.names.index('agent0/shell')
        self.geom_id['ball2/shell'] = physics.named.model.geom_bodyid.axes.row.names.index('ball2/shell')
        self.geom_id['ball3/shell'] = physics.named.model.geom_bodyid.axes.row.names.index('ball3/shell')
        self.geom_id['ball4/shell'] = physics.named.model.geom_bodyid.axes.row.names.index('ball4/shell')
        self.geom_id['ball5/shell'] = physics.named.model.geom_bodyid.axes.row.names.index('ball5/shell')
        self.geom_id['target/geom'] = physics.named.model.geom_bodyid.axes.row.names.index('target/geom')

        self.collision_tracker = pd.DataFrame(data=0, index=[0], columns=self.geom_id.keys())
        self.episode_data = []

    #def _safe_pixel(self, point, max_x, max_y):
    #    return np.clip(point[0],0,max_x), np.clip(point[1],0,max_y)

    def _create_image_directory(self):
        os.mkdir(self.image_directory)

    def _clear_image_directory(self):
        fp = self.image_directory+'*'
        files = glob.glob(fp)

        for file in files:
            os.remove(file)

    def update_obs_count(self,
                         raw_obs) -> None:

        if self.env_name == 'FlatlandMultiagentZfn-v0':
            for i in range(len(raw_obs['a'])):
                x, y = self._raw_to_pixel(raw_obs['a'][i])
                self.plots['raw_state_count'][x][y] += 1
                self.plots['raw_episode_state_count'][x][y] += 1

    @staticmethod
    def _clip(input, lower, upper):
        return max(min(input, upper), lower)

    def write_images(self,
                     image_num: int) -> None:

        if self.env_name == 'FlatlandMultiagentZfn-v0':

            # Output count-related images

            plots_to_write = [{'metric': 'raw_state_count',
                               'normalization': 'log'},

                              {'metric': 'raw_episode_state_count',
                               'normalization': 'log'}]

            for plot in plots_to_write:
                self._write_image(pixels=self._array_to_pixels(self.plots[plot['metric']],
                                                               normalization=plot['normalization']),
                                  name=plot['metric'],
                                  image_num=image_num)


        self._reset_after_writing()

    def _reset_after_writing(self):

        self.plots['raw_episode_state_count'] = np.zeros(self.shape, dtype=np.int)

    @staticmethod
    def _array_to_pixels(array: np.ndarray,
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
            normed_array = np.power(array, (1.0/3.0))
            pixels = 255 - np.minimum(normed_array * 200, 255)

        elif normalization == "linear-shift-value":
            normed_array = array + 20
            pixels = 255 - np.maximum(np.minimum(normed_array, 255),0)

        elif normalization == "none":
            pixels = 255 - array

        else:
            raise ValueError('Unknown normalization: ' + normalization)

        # Convert type
        return pixels.astype('uint8')

    def _write_image(self,
                     pixels: np.ndarray,
                     name: str,
                     image_num: int) -> None:

        im = Image.fromarray(pixels)
        im.save(self.image_directory + name + '_' + str(image_num).zfill(4) + '.png')

    def create_gifs(self) -> None:
        # TODO: add logging of images to WandB
        if self.env_name == 'FlatlandMultiagentZfn-v0':
            # filepaths
            names = ['raw_state_count',
                     'raw_episode_state_count',
                     'intrinsic_reward']

            for name in names:
                self._create_gif(self.image_directory + name + '_*.png',
                                 self.image_directory + 'GIF_' + name + '.gif',
                                 self.image_directory + 'SNAP_' + name + '.png')

    @staticmethod
    def _create_gif(fp_in: str,
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

