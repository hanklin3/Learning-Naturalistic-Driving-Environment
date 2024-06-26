# %%
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import xml.etree.ElementTree as ET
# from shapely.geometry import Point
import os
# import shutil
from PIL import Image, ImageDraw 
from vehicle import Vehicle
import pickle

# %%

base_dir = '/mnt/d/OneDrive - Massachusetts Institute of Technology/Research/sumo/'
base_dir = './data/sumo/'
assert os.path.exists(base_dir)

# simulated_file = os.path.join(base_dir, "ring_faster/out.xml")
simulated_file = os.path.join(base_dir, "ring/out.xml")

assert os.path.exists(simulated_file)


# %%
tree = ET.parse(simulated_file)

print(tree)
sim_root = tree.getroot()

print(sim_root)

sim_data = []
for child in sim_root:
    print(child)
    print('child.attrib', child.attrib)
    for child2 in child:
        print(child2)
        if child2.tag == 'vehicle':
            print(child2.attrib)
            rowd = child2.attrib
            rowd['time'] = child.attrib['time']
            sim_data.append(child2.attrib)
            print('child2.attrib', child2.attrib)
sim_df = pd.DataFrame(sim_data)

# %%
sim_df

# %%
negY = -95.09901674512636 
posX = 170.48733984863466
offsetY = negY
multiplier = (abs(negY))/80
to_zero = 1.6
offsetY = -80
multiplier = 1

xs = [(float(x)) + to_zero for x in sim_df['x']]
ys = [(float(y)) + to_zero for y in sim_df['y']]

sim_df['x'] = [(float(x) + to_zero)*multiplier  for x in sim_df['x']]
sim_df['y'] = [(float(y) + to_zero)*multiplier + offsetY for y in sim_df['y']]

sim_df['time'] = sim_df['time'].astype(float)
sim_df['x'] = sim_df['x'].astype(float)
sim_df['y'] = sim_df['y'].astype(float)
sim_df['id'] = sim_df['id'].astype(int)
sim_df['angle'] = sim_df['angle'].astype(float)
sim_df['speed'] = sim_df['speed'].astype(float)

plt.scatter(sim_df['x'],sim_df['y'])

minx, maxx = min(sim_df['x']), max(sim_df['x'])
miny, maxy = min(sim_df['y']), max(sim_df['y'])
print('min/max x/y', minx, maxx, miny, maxy)
print('id', sim_df['id'])

print('max time', max(sim_df['time']))

# %%

def get_veh_list(t):
    init_df = sim_df[sim_df['time']==t]
    xs = init_df['x'].tolist()
    ys = init_df['y'].tolist()
    ids = init_df['id'].tolist()
    angles = init_df['angle'].tolist()
    speeds = init_df['speed'].tolist()

    print(xs)

    v = Vehicle()
    print(v.location.x, v.location.y)

    # change to 0.4s

    vehicle_list = []
    for i in range(len(xs)):
        v = Vehicle()
        v.location.x = xs[i]
        v.location.y = ys[i]
        v.id = ids[i]
        # sumo: north, clockwise
        # yours: east, counterclockwise
        v.speed_heading = (-angles[i] + 90 ) % 360
        v.speed = speeds[i]

        factor = 1
        v.size.length, v.size.width = 3.6*factor, 1.8*factor
        v.safe_size.length, v.safe_size.width = 3.8*factor, 2.0*factor
        v.update_poly_box_and_realworld_4_vertices()
        v.update_safe_poly_box()
        vehicle_list.append(v)


    if not vehicle_list:
        print('t', t)
        print(init_df)

    print('t', t)
    v = vehicle_list[0]
    print('x, y, heading', v.location.x, v.location.y, v.id, v.speed_heading)
    print('poly_box', vars(v.poly_box))
    print('v.size.length, v.size.width', v.size.length, v.size.width)
    return vehicle_list


def init_pickle(vehicle_list):
    path = './data/inference/ring/simulation_initialization/gen_veh_states/ring'
    output_file_path = os.path.join(path, 'initial_vehicle_dict.pickle')
    vehicle_dict = {'n_in1': vehicle_list}
    pickle.dump(vehicle_dict, open(output_file_path, "wb"))

# %%
t = 0.0
count = 0
while t < 1000.0:
    t = round(t, 2)

    vehicle_list = get_veh_list(t)
    if t==0:
        init_pickle(vehicle_list)
        t += 0.4
        count += 1
        continue
    
    path = './data/inference/ring/simulation_initialization/initial_clips/ring-01/01/'
    output_file_path = os.path.join(path,f"{count:02d}.pickle")
    print('output_file_path', output_file_path)
    pickle.dump(vehicle_list, open(output_file_path, "wb"))

    path = './data/training/behavior_net/ring/ring257/train/01/01'
    output_file_path = os.path.join(path,f"{count:02d}.pickle")
    print('output_file_path', output_file_path)
    pickle.dump(vehicle_list, open(output_file_path, "wb"))

    if count < 200:
        path = './data/training/behavior_net/ring/ring257/val/01/01'
        output_file_path = os.path.join(path,f"{count:02d}.pickle")
        print('output_file_path', output_file_path)
        pickle.dump(vehicle_list, open(output_file_path, "wb"))

    t += 0.4
    count += 1


# %%
t=6
vehicle_list = get_veh_list(t)
vxs = [v.location.x for v in vehicle_list]
vys = [v.location.y for v in vehicle_list]
plt.plot(vxs, vys, '.')
plt.axis('equal')

# %%
# img_size = (round(maxx)+1, round(maxy)+1, 3)
img_size = (80,80, 3)
img_resize = (200, 200)
# img_resize = img_size
img = np.zeros(img_size)
# for x, y in zip(sim_df['x'], sim_df['y']):
for x, y in zip(xs, ys):
    img[int(x+to_zero), int(y+to_zero), :] = 255
img = img.astype(np.uint8)
img = cv2.resize(img, img_resize)
ksize = (15, 15) 
# Using cv2.blur() method  
img = cv2.blur(img, ksize)  
img[img > 0] = 255

plt.imshow(img)
print(img.min(), img.max(), np.unique(img))

# %%
base_dir = './data/inference/ring/'
save_img_path = os.path.join(base_dir,'drivablemap', 'ring-drivablemap.jpg')
plt.imsave(save_img_path, img)
save_img_path = os.path.join(base_dir,'basemap', 'ring-official-map.jpg')
plt.imsave(save_img_path, img)
for filename in ['circle_1_q-map.jpg', 'circle_2_q-map.jpg', 'circle_3_q-map.jpg', 'circle_4_q-map.jpg']:
    save_img_path = os.path.join(base_dir,'ROIs-map','circle', filename)
    plt.imsave(save_img_path, img)

for filename in ['circle_inner_lane-map.jpg','circle_outer_lane-map.jpg'
                 ]:
    save_img_path = os.path.join(base_dir,'ROIs-map','at-circle-lane', filename)
    plt.imsave(save_img_path, img)
print('save_img_path', save_img_path)

# %%
json_text = f'"tl": [{minx}, {maxy}], "bl": [{minx}, {miny}], "tr": [{maxx}, {maxy}], "br": [{maxx}, {miny}]'
json_text = '{' + json_text + '}'
print(json_text)
json_text = f'"tl": [{0.0}, {img_size[1]}], "bl": [{0.0}, {0.0}], "tr": [{img_size[0]}, {img_size[1]}], "br": [{img_size[0]}, {0.0}]'
json_text = '{' + json_text + '}'
print(json_text)

# %%
empty_img = np.zeros(img_resize)
save_img_path = os.path.join(base_dir,'ROIs-map', 'ring-sim-remove-vehicle-area-map.jpg')
plt.imsave(save_img_path, empty_img)
print('save_img_path', save_img_path)

for filename in ['exit_n-map.jpg','exit_e-map.jpg', 'exit_s-map.jpg',
                 'exit_w-map.jpg', 'exit_n_rightturn-map.jpg', 
                 'exit_s_rightturn-map.jpg'
                 ]:
    save_img_path = os.path.join(base_dir,'ROIs-map','exit', filename)
    plt.imsave(save_img_path, empty_img)
for filename in ['entrance_n_1-map.jpg','entrance_n_2-map.jpg', 'entrance_n_rightturn-map.jpg',
                 'entrance_e_1-map.jpg', 'entrance_e_2-map.jpg', 
                 'entrance_s_1-map.jpg', 'entrance_s_2-map.jpg',
                 'entrance_s_rightturn-map.jpg', 
                 'entrance_w_1-map.jpg', 'entrance_w_2-map.jpg'
                 ]:
    save_img_path = os.path.join(base_dir,'ROIs-map','entrance', filename)
    plt.imsave(save_img_path, empty_img)
for filename in ['yielding_n-map.jpg','yielding_e-map.jpg', 'yielding_s-map.jpg',
                 'yielding_w-map.jpg'
                 ]:
    save_img_path = os.path.join(base_dir,'ROIs-map','yielding-area', filename)
    plt.imsave(save_img_path, empty_img)

# pos = [float(x) for x in  sim_df['pos'][0].split()]
# x, y, z = pos
# print(pos)

# positions = []
# for index, row in sim_df.iterrows():
#     positions.append([float(x) for x in  row['pos'].split()])

# print(positions)
# positions = np.asarray(positions)
# print(positions.shape, positions)

# %%
plt.imshow(img)
img2 = Image.fromarray(img)
img3 = ImageDraw.Draw(img2)
# %%
img3 = ImageDraw.Draw(img2)
# %%
plt.imshow(np.asarray(img3))
# %%
img = np.zeros(img_size)
plt.imsave(os.path.join(base_dir, 'ring-sim-remove-vehicle-area-map.jpg'), img)
# %%
