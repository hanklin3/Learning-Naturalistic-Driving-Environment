# %%
import cv2
import glob
import os
import pickle
from pprint import pprint
import matplotlib.pyplot as plt

# %%
# from simulation_modeling.simulation_inference import save_time_buff_video
from simulation_modeling.trajectory_interpolator import TrajInterpolator
from basemap import Basemap
import yaml

config = './configs/AA_rdbt_inference.yml'
config = './configs/ring_inference.yml'


# %%
path = './data/inference/rounD/simulation_initialization/initial_clips/rounD-09/01'
path = './data/training/behavior_net/AA_rdbt/AA_rdbt-10h-data-local-heading-size-36-18/train/01/02/'
path = './data/training/behavior_net/ring/ring257/train/01/01/'
path = './data/training/behavior_net/ring_larger/ring257/train/01/01/'
path = './data/inference/ring_larger/simulation_initialization/initial_clips/ring-01/01'
config = './configs/ring_inference_larger.yml'

# path = './data/inference/ring/simulation_initialization/initial_clips/ring-01/01'
# path = './results/inference/ring_inference/ring/300s/TIME_BUFF/3/0/'

configs = yaml.safe_load(open(config))
print(f"Loading config file: {config}")

def get_vehicle_list(max_timestep=0):
    TIME_BUFF=[]
    datafolder_dirs = glob.glob(os.path.join(path, '*.pickle'))
    datafolder_dirs.sort()
    print('datafolder_dirs', datafolder_dirs)
    for step, datafolder_dir in enumerate(datafolder_dirs):
        print('datafolder_dir', datafolder_dir)
        vehicle_list = pickle.load(open(datafolder_dir, "rb"))
        TIME_BUFF.append(vehicle_list)
        if max_timestep > 0 and step > max_timestep:
            break
    return TIME_BUFF

# %%

TIME_BUFF = get_vehicle_list(max_timestep=0)

# TIME_BUFF=[]
# TIME_BUFF.append(vehicle_list)

print(TIME_BUFF[0])

# %%
# for i in range(len(TIME_BUFF)):
i=0
for j in range(len(TIME_BUFF[i])):
    v = TIME_BUFF[i][j]
    print('v', v)
    print('x, y, id', v.location.x, v.location.y, v.id)

# %%
# plot_save_path = './data/inference/ring/imgs_plot'
# for i in range(len(TIME_BUFF)):
#     vehicle_list = TIME_BUFF[i]
#     vxs = [v.location.x for v in vehicle_list]
#     vys = [v.location.y for v in vehicle_list]
#     ids = [str(v.id) for v in vehicle_list]
#     plt.figure()
#     plt.plot(vxs, vys, '.')
#     for iid in range(len(ids)):
#         plt.text(vxs[iid], vys[iid], ids[iid])
#     plt.axis('equal')
#     plt.savefig(f'{plot_save_path}/{i:02d}.jpg')
#     if i % 10 == 0:
#         print('Saved to', f'{plot_save_path}/{i:02d}.jpg')
# %%

background_map = Basemap(map_file_dir=configs["basemap_dir"], map_height=configs["map_height"], map_width=configs["map_width"])

def _visualize_time_buff(TIME_BUFF, background_map, i = 4):
    # for i in range(len(TIME_BUFF)):
    
    vehicle_list = TIME_BUFF[i]
    vis = background_map.render(vehicle_list, with_traj=True, linewidth=3)
    img = vis[:, :, ::-1]
    # plt.imshow(img)
    return img
    
# %%
# debug_img_path = './data/inference/ring/imgs'
# for i in range(len(TIME_BUFF)):
#     img = _visualize_time_buff(TIME_BUFF, background_map, i)
#     cv2.imwrite(f'{debug_img_path}/{i}.jpg', img)
# %%
def save_time_buff_video(TIME_BUFF, background_map, file_name, 
                         save_path, color_vid_list=None, with_traj=True,
                         interpolate_flag=True):
    traj_interpolator = TrajInterpolator()
    if interpolate_flag:
            visualize_TIME_BUFF, _ = traj_interpolator.interpolate_traj(TIME_BUFF,
                                        intep_steps=configs["intep_steps"],
                                        original_resolution=configs["sim_resol"])
    else:
        visualize_TIME_BUFF = TIME_BUFF

    save_fps = configs["save_fps"]
    print('saved fps', save_fps)
    os.makedirs(save_path, exist_ok=True)
    saved_file_mp4 = save_path + r'/{0}.mp4'.format(file_name)
    print('Saving to...', saved_file_mp4)
    collision_video_writer = cv2.VideoWriter(saved_file_mp4, cv2.VideoWriter_fourcc(*'MP4V'), save_fps, (background_map.w, background_map.h))
    for i in range(len(visualize_TIME_BUFF)):
        vehicle_list = visualize_TIME_BUFF[i]
        vis = background_map.render(vehicle_list, with_traj=with_traj, linewidth=6, color_vid_list=color_vid_list)
        img = vis[:, :, ::-1]
        # img = cv2.resize(img, (768, int(768 * background_map.h / background_map.w)))  # resize when needed
        collision_video_writer.write(img)
    print('Saved to', saved_file_mp4)

# save_path = './data/inference/ring'
save_path = path
with_traj = True
if with_traj:
    filename = 'ring_step_0.4s_wTrail_no_jump'
else:
    filename = 'ring_step_0.4s_noTrail_no_jump'
save_time_buff_video(TIME_BUFF, background_map, filename, save_path, with_traj=with_traj,
                     interpolate_flag=False)

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

path = './data/inference/rounD/simulation_initialization/gen_veh_states/rounD/'
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

