# %%
import cv2
import glob
import os
import numpy as np
import pandas as pd
import pickle
from pprint import pprint
import matplotlib.pyplot as plt

# %%
from simulation_modeling.simulation_inference import SimulationInference
from sim_evaluation_metric.realistic_metric import RealisticMetrics

from basemap import Basemap
import yaml

config = './configs/AA_rdbt_inference.yml'
config = './configs/ring_inference.yml'

configs = yaml.safe_load(open(config))
print(f"Loading config file: {config}")

# %%
path = './data/inference/rounD/simulation_initialization/initial_clips/rounD-09/01'
path = './data/training/behavior_net/AA_rdbt/AA_rdbt-10h-data-local-heading-size-36-18/train/01/02/'
path = './data/training/behavior_net/ring/ring257/train/01/01/'
# path = './data/inference/ring/simulation_initialization/initial_clips/ring-01/01/'
path = './results/inference/ring_inference/ring_0.4_200x200/36s/TIME_BUFF/1/0'
# path = './data/inference/ring/simulation_initialization/initial_clips/ring-01/01'
path = './data/inference/ring/simulation_initialization/initial_clips/ring-01/01'
path = './results/inference/ring_inference/ring/36s/TIME_BUFF/1/0'
path = './results/inference/ring_inference/ring/300s/TIME_BUFF/3/0'
path = './results/inference/ring_inference/ring_0.4_no_jump/300s/TIME_BUFF/3/0'
path = './results/inference/ring_inference/ring_0.4_no_jump2/300s/TIME_BUFF/1/0'
path = './results/inference/ring_inference/ring_0.4_no_jump2/300s/TIME_BUFF/2/0'
path = './results/inference/ring_inference/ring_0.4_no_jump2/600s/TIME_BUFF/3/0'
path = './results/inference/ring_inference/ring_0.4_no_jump2/600s/TIME_BUFF/4/0'
path = './results/inference/ring_inference/ring_0.4_no_jump2/300s/TIME_BUFF/5/0'
path = './data/inference/ring/simulation_initialization/initial_clips/ring-01/01'

def get_vehicle_list(max_timestep=0):
    TIME_BUFF=[]
    datafolder_dirs = sorted(glob.glob(os.path.join(path, '*.pickle')))
    print('datafolder_dirs', datafolder_dirs)
    for step, datafolder_dir in enumerate(datafolder_dirs):
        vehicle_list = pickle.load(open(datafolder_dir, "rb"))
        TIME_BUFF.append(vehicle_list)
        if max_timestep > 0 and step > max_timestep:
            break
    return TIME_BUFF

# %%
max_timestep = 200
# max_timestep = 500
# max_timestep = 0
remove_phantom_cars = False
TIME_BUFF = get_vehicle_list(max_timestep=max_timestep)

print(len(TIME_BUFF))
print(TIME_BUFF[-1])

# %%
########### Creating df for time-space diagram ##################

time_increment = 0.4
dataf = []
for i in range(len(TIME_BUFF)):
    for j in range(len(TIME_BUFF[i])):
        v = TIME_BUFF[i][j]
        print('v', v)
        print('x, y, id', v.location.x, v.location.y, v.id)
        time = i*time_increment
        # Remove phantom cars
        if remove_phantom_cars and int(v.id) > 21:
            continue
        dataf.append([int(0), time, int(v.id), 
                      float(v.location.x), float(v.location.y)])
        


# %%
arr = np.asarray(dataf)
df_traj = pd.DataFrame(arr,
                       columns=['Simulation No', 'Time', 'Car', 'x', 'y'])
df_traj['Simulation No'] = df_traj['Simulation No'].astype(int)
df_traj['Car'] = df_traj['Car'].astype(int)
df_traj
# %%
print(np.unique(df_traj['Car']))
# %%
df = df_traj[df_traj['Car']==13]
df
# %%
# plt.plot(df['x'].values, df['y'].values, '.')
plt.scatter(df['x'].values, df['y'].values)
plt.gca().set_aspect("equal")
# %%
df = df_traj[df_traj['Time']==12]
df
# %%
##################Compute distance by circle arc ###############################
dia =  np.max(df_traj['x']) - np.min(df_traj['x'])
dia2 =  np.max(df_traj['y']) - np.min(df_traj['y'])
dia = max(dia,dia2)
r = dia/2
print('dia', dia, 'radius', r)
# %%
Xstart, Ystart = (np.min(df_traj['x']) + np.max(df_traj['x']))/2, np.min(df_traj['y'])
Xstart, Ystart
# %%
def dist(x1,x2, y1,y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def get_pos(x, y):
    d = dist(Xstart,x, Ystart, y)
    # print('1-d**2/2*r**2', 1 - d**2/(2*r**2))
    rad = np.arccos(1 - d**2/(2*r**2))

    
    length = r * rad
    if np.isnan(length):
        print('nan',x, y)
        print('1-d**2/2*r**2', 1 - d**2/(2*r**2))
        print('rad', rad)
    return length

def get_pos2(x, y):
    rad = np.arctan2(y-Ystart, x-Xstart)
    print('rad', rad)
    rad[rad<0] = rad[rad<0] + 2*np.pi
    # if rad < 0:
    #     rad += 2*np.pi
    length = r * rad
    return length

# %%
for index, row in df.iterrows():
    #print(row['x'], row['y'])
    # pos = get_pos(row['x'], row['y'])
    # print('pos', get_pos(row['x'], row['y']))
    print('pos2', get_pos2(np.asarray([row['x']]), np.asarray([row['y']])))
    # break
# %%
poss = get_pos2(df['x'].values, df['y'].values)
poss
# %%
np.arccos([1, -1])
# %%
poss = get_pos2(df_traj['x'].values, df_traj['y'].values)
poss
#######################################################
# %%
df_traj.insert(len(df_traj.columns), "Position", poss)
df_traj
# %%
df_traj[df_traj['Car']==21]['Position']
# %%
print(np.unique(df_traj['Car']))
# %%
save_path = os.path.join(path, f'df_traj_{max_timestep}.csv')
df_traj.to_csv(save_path)
print('saved to ', save_path)

# %%
len(TIME_BUFF)

# %%
print('Time-space diagram DONE!!')
assert False
# %%

########### Generating groundtruth distribution ##################
save_result_path = './data/statistical-realism-ground-truth/ring_ground_truth'
configs["realistic_metric_save_folder"] = os.path.join(save_result_path)
configs["simulated_TIME_BUFF_save_folder"] = os.path.join(save_result_path, 'simulation')
configs["save_viz_folder"] = os.path.join(save_result_path, 'viz')
configs["device"] = 'cuda:0'

simulation_inference_model = SimulationInference(configs=configs)
simulation_inference_model.generate_simulation_metric(TIME_BUFF)
simulation_inference_model.save_sim_metric()
print('Saved to', configs["realistic_metric_save_folder"])
# %%
simulation_inference_model.save_sim_metric()
print('Saved to', configs["realistic_metric_save_folder"])
# %%
simulation_inference_model._gen_sim_metric(TIME_BUFF)

# %%
# ROIs
# circle_map_dir = os.path.join(configs["ROI_map_dir"], 'circle')
# entrance_map_dir = os.path.join(configs["ROI_map_dir"], 'entrance')
# exit_map_dir = os.path.join(configs["ROI_map_dir"], 'exit')
# crosswalk_map_dir = None  # os.path.join(configs["ROI_map_dir"], 'crosswalk')
# yielding_area_map_dir = os.path.join(configs["ROI_map_dir"], 'yielding-area')
# at_circle_lane_map_dir = os.path.join(configs["ROI_map_dir"], 'at-circle-lane')


# basemap_img = cv2.imread(configs["basemap_dir"], cv2.IMREAD_COLOR)
# basemap_img = cv2.cvtColor(basemap_img, cv2.COLOR_BGR2RGB)
# basemap_img = cv2.resize(basemap_img, (configs["map_width"], configs["map_height"]))
# basemap_img = (basemap_img.astype(np.float64) * 0.6).astype(np.uint8)

# PET_configs = configs["PET_configs"]
# PET_configs["basemap_img"] = basemap_img

# SimMetricsAnalyzer = RealisticMetrics(drivable_map_dir=configs["drivable_map_dir"], sim_remove_vehicle_area_map=configs["sim_remove_vehicle_area_map"],
#                                                        circle_map_dir=circle_map_dir, entrance_map_dir=entrance_map_dir, exit_map_dir=exit_map_dir,
#                                                        crosswalk_map_dir=crosswalk_map_dir, yielding_area_map_dir=yielding_area_map_dir, at_circle_lane_map_dir=at_circle_lane_map_dir,
#                                                        sim_resol=configs["sim_resol"],
#                                                        map_height=configs["map_height"], map_width=configs["map_width"],
#                                                        PET_configs=PET_configs)
# # %%
# SimMetricsAnalyzer.construct_traj_data(TIME_BUFF)
# instant_speed_list = SimMetricsAnalyzer.in_circle_instant_speed_analysis()
# %%

# %%
