# %%
import cv2
import glob
import os
import pickle
from pprint import pprint

print('here')

# %%
path = '../data/inference/rounD/simulation_initialization/initial_clips/rounD-09/01'
path = '../data/training/behavior_net/AA_rdbt/AA_rdbt-10h-data-local-heading-size-36-18/train/01/02/'
datafolder_dirs = glob.glob(os.path.join(path, '*.pickle'))
print('datafolder_dirs', datafolder_dirs)
vehicle_list = pickle.load(open(datafolder_dirs[0], "rb"))


# %%
vehicle_list
TIME_BUFF=[]
TIME_BUFF.append(vehicle_list)
print(TIME_BUFF[:10])

for i in range(len(TIME_BUFF)):
    for j in range(len(TIME_BUFF[i])):
        v = TIME_BUFF[i][j]
        print('v', v)
        print('x, y, id', v.location.x, v.location.y, v.id)
# %%

path = '../data/inference/rounD/simulation_initialization/gen_veh_states/rounD/'
datafolder_dirs = glob.glob(os.path.join(path, '*.pickle'))
vehicle_list = pickle.load(open(datafolder_dirs[0], "rb"))
vehicle_list.keys()

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
