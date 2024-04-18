# %%
import cv2
import glob
import os
import pickle
from pprint import pprint
import matplotlib.pyplot as plt

# %%
# from simulation_modeling.simulation_inference import _visualize_time_buff

print('here')

# %%
path = './data/inference/rounD/simulation_initialization/initial_clips/rounD-09/01'
path = './data/training/behavior_net/AA_rdbt/AA_rdbt-10h-data-local-heading-size-36-18/train/01/02/'
path = './data/training/behavior_net/ring/ring257/train/01/01/'
# path = './data/inference/ring/simulation_initialization/initial_clips/ring-01/01/'
datafolder_dirs = glob.glob(os.path.join(path, '*.pickle'))
print('datafolder_dirs', datafolder_dirs)
vehicle_list = pickle.load(open(datafolder_dirs[0], "rb"))
vehicle_list

# %%

TIME_BUFF=[]
TIME_BUFF.append(vehicle_list)
print(TIME_BUFF[:10])

for i in range(len(TIME_BUFF)):
    for j in range(len(TIME_BUFF[i])):
        v = TIME_BUFF[i][j]
        print('v', v)
        print('x, y, id', v.location.x, v.location.y, v.id)


# %%
from basemap import Basemap
import yaml

config = './configs/AA_rdbt_inference.yml'
config = './configs/ring_inference.yml'


configs = yaml.safe_load(open(config))
print(f"Loading config file: {config}")


background_map = Basemap(map_file_dir=configs["basemap_dir"], map_height=configs["map_height"], map_width=configs["map_width"])

def _visualize_time_buff(TIME_BUFF, background_map):
    for i in range(len(TIME_BUFF)):
        vehicle_list = TIME_BUFF[i]
        vis = background_map.render(vehicle_list, with_traj=True, linewidth=6)
        img = vis[:, :, ::-1]
        plt.imshow(img)

_visualize_time_buff(TIME_BUFF, background_map)

# %%
# # Demo of our trajectory format
# import pickle
# import os
# import glob

# def load_traj(one_video):
#     TIME_BUFF = []
#     for i in range(0, len(one_video)):
#         vehicle_list = pickle.load(open(one_video[i], "rb"))
#         TIME_BUFF.append(vehicle_list)
#     print("Trajectory length: {0} s".format(len(TIME_BUFF) * 0.4))
#     return TIME_BUFF

# # Load all frames of a episode
# one_video = glob.glob(
#     os.path.join(r'./results/inference/AA_rdbt_inference/example/traj_data', '*.pickle'))
# one_sim_TIME_B
# %%

path = '../data/inference/rounD/simulation_initialization/gen_veh_states/rounD/'
datafolder_dirs = glob.glob(os.path.join(path, '*.pickle'))
vehicle_list = pickle.load(open(datafolder_dirs[0], "rb"))
vehicle_list.keys()
vehicle_list['n_in1']


# %%

pprint(vars(v))

# %%
print('location', vars(v.location))
print('gt_size', vars(v.gt_size))
print('pixel_bottom_center', vars(v.pixel_bottom_center))
print('poly_box', vars(v.poly_box))
print('rotation', vars(v.rotation))
print('safe_poly_box', vars(v.safe_poly_box))
print('safe_size', vars(v.safe_size))
print('size', vars(v.size))

# %%
path = '../data/inference/rounD/simulation_initialization/gen_veh_states/rounD/'
datafolder_dirs = glob.glob(os.path.join(path, '*.pickle'))
vehicle_list = pickle.load(open(datafolder_dirs[0], "rb"))
vehicle_list

# %%
map_file_dir = '../data/inference/rounD/basemap/rounD-official-map.png'
basemap = cv2.imread(map_file_dir, cv2.IMREAD_COLOR)
print(basemap.shape)

# %%
map_file_dir = '../data/inference/rounD/drivablemap/rounD-drivablemap.jpg'
basemap = cv2.imread(map_file_dir, cv2.IMREAD_COLOR)
print(basemap.shape, basemap.min(), basemap.max())
# %%
map_file_dir = '../data/sumo/ring-drivablemap.jpg'
basemap = cv2.imread(map_file_dir, cv2.IMREAD_COLOR)
print(basemap.shape, basemap.min(), basemap.max())
plt.imshow(basemap)

# %%

