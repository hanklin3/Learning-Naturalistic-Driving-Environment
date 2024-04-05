# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xml.etree.ElementTree as ET
# from shapely.geometry import Point
import os
# import shutil
from PIL import Image, ImageDraw 
from vehicle import Vehicle
# import pickle

# %%

base_dir = '/mnt/d/OneDrive - Massachusetts Institute of Technology/Research/sumo/'
base_dir = '../data/sumo/'
assert os.path.exists(base_dir)

simulated_file = os.path.join(base_dir, "ring/out.xml")

assert os.path.exists(simulated_file)

# %%
tree = ET.parse(simulated_file)
# %%
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
sim_df = pd.DataFrame(sim_data)

# %%
sim_df

# %%
offset = 2
xs = [float(x) + offset for x in sim_df['x']]
ys = [float(y) + offset for y in sim_df['y']]
ids = [int(id) + offset for id in sim_df['id']]

plt.scatter(xs, ys)

print(min(xs), max(xs), min(ys))
print('id', ids)

# %%
v = Vehicle()
print(v.location.x, v.location.y)

vehicle_list = []
for i in range(2):
    v = Vehicle()
    v.location.x = xs[i]
    v.location.y = ys[i]
    v.id = ids[i]

    if ys[i] > 70 or ys[i] < 10:
        vehicle_list.append(v)

v = vehicle_list[0]
print(v.location.x, v.location.y, v.id)

# %%
path = '../data/inference/ring/simulation_initialization/gen_veh_states/ring'
output_file_path = os.path.join(path, 'initial_vehicle_dict.pickle')
vehicle_dict = {'n_in1': vehicle_list}
pickle.dump(vehicle_dict, open(output_file_path, "wb"))
# %%
img = np.zeros((80,80, 3))
for x, y in zip(xs, ys):
    img[int(x), int(y), :] = 255
img = img.astype(np.uint8)

plt.imshow(img)
print(img.min(), img.max(), np.unique(img))
# %%
plt.imsave(os.path.join(base_dir, 'ring-drivablemap.jpg'), img)

# %%
plt.imsave(os.path.join(base_dir, 'ring-sim-remove-vehicle-area-map.jpg'), np.zeros((80,80, 3)))

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
img = np.zeros((80,80))
plt.imsave(os.path.join(base_dir, 'ring-sim-remove-vehicle-area-map.jpg'), img)
# %%
