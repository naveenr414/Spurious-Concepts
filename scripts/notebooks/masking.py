# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: cem
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

import sys
sys.path.append('/usr0/home/naveenr/projects/spurious_concepts/ConceptBottleneck/')
sys.path.append('/usr0/home/naveenr/projects/spurious_concepts')

import torch
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
import torch.nn as nn
import torch.optim as optim
import pickle
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap
import cv2
from copy import copy 
import itertools
from matplotlib.patches import Circle
import json
import argparse
import logging 
import resource
import gc 

from ConceptBottleneck.CUB.dataset import load_data

from src.images import *
from src.util import *
from src.models import *
from src.plot import *

# ## Set up dataset + model

torch.cuda.set_per_process_memory_fraction(0.5)
resource.setrlimit(resource.RLIMIT_AS, (30 * 1024 * 1024 * 1024, -1))
torch.set_num_threads(1)

logging.basicConfig(level=logging.INFO)
logging.info("Setting up dataset")

# +
is_jupyter = 'ipykernel' in sys.modules
if is_jupyter:
    encoder_model='inceptionv3'
    seed = 44
    dataset_name = "CUB"
else:
    parser = argparse.ArgumentParser(description="Synthetic Dataset Experiments")

    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()
    encoder_model = "inceptionv3" 
    seed = args.seed 
    dataset_name = "CUB"

parameters = {
    'dataset': dataset_name,
    'seed': seed, 
    'encoder_model': encoder_model ,
    'debugging': False,
    'epochs': 100,
    'lr': 0.005,
}

# -

train_loader, val_loader, test_loader, train_pkl, val_pkl, test_pkl = get_data(1,encoder_model=encoder_model,dataset_name=dataset_name)

test_images, test_y, test_c = unroll_data(test_loader)

log_folder = get_log_folder(dataset_name,parameters).split("/")[-1]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

joint_model = get_synthetic_model(dataset_name,parameters)

joint_model = joint_model.to(device)

run_model_function = run_joint_model

# ## Plot the Dataset

logging.info("Plotting Dataset")

dataset_directory = "../../../../datasets"

img_path = dataset_directory+'/'+train_pkl[0]['img_path']
image = Image.open(img_path)
plt.imshow(image)

# ## Using Part Annotations

logging.info("Using part annotations")

# +
part_location = dataset_directory + "/CUB/metadata/parts/part_locs.txt"
part_list = dataset_directory + "/CUB/metadata/parts/parts.txt"

part_file = open(part_location).read().strip().split("\n")
part_list = open(part_list).read().strip().split("\n")
part_list = [' '.join(i.split(' ')[1:]) for i in part_list]

attribute_names = open(dataset_directory+"/CUB/metadata/attributes.txt").read().strip().split("\n")
attribute_names = [' '.join(i.split(' ')[1:]) for i in attribute_names]

# +
parts_to_attribute = {}

for i in range(len(part_list)):
    if 'left' in part_list[i] or 'right' in part_list[i]:
        opposite = part_list[i].replace('left','RIGHT').replace('right','LEFT').lower()
        indices = sorted([i,part_list.index(opposite)])
        current_name = '^'.join([str(j) for j in indices])
    else:
        current_name = str(i)
    
    parts_to_attribute[current_name] = [] 
    parts_split = part_list[i].split(' ')

    for j in range(len(attribute_names)):
        main_part = set(attribute_names[j].split("::")[0].split("_"))

        if len(main_part.intersection(parts_split)) > 0:
            parts_to_attribute[current_name].append(j)
# -

locations_by_image_id = {}
for i in part_file:
    split_vals = i.split(' ')

    if split_vals[-1] == '1':
        image_id = int(split_vals[0])
        part_id = int(split_vals[1])-1 # 0 index 
        x = float(split_vals[2])
        y = float(split_vals[3])

        if image_id not in locations_by_image_id:
            locations_by_image_id[image_id] = {}
        locations_by_image_id[image_id][part_id] = (x,y)

# #### Impact of Masking on Predictions

logging.info("Impact of Epsilon")

with torch.no_grad():
    initial_predictions = [] 

    for data_point in test_loader:
        x,y,c = data_point 
        _, initial_predictions_batch = run_joint_model(joint_model,x.to(device))
        initial_predictions_batch = torch.nn.Sigmoid()(initial_predictions_batch.detach().cpu().T)
        initial_predictions.append(initial_predictions_batch.numpy())
    initial_predictions = np.row_stack(initial_predictions)
    

torch.cuda.empty_cache()

valid_parts = [int(i) for i in parts_to_attribute if '^' not in i and parts_to_attribute[i] != []]

main_part = 13
test_data_num = 100
valid_data_points = [i for i in range(len(test_pkl)) if main_part in locations_by_image_id[test_pkl[i]['id']]]
data_points = random.sample(valid_data_points,test_data_num)

epsilon_list = [0.01,0.02,0.05,0.1,0.15,0.2,0.3]

# +
diff_by_epsilon = {}

for epsilon in epsilon_list:
    print("Epsilon is {}".format(epsilon))
    masked_images = torch.stack([mask_image_location(test_images[data_point], get_part_location(data_point,main_part, locations_by_image_id, test_pkl), color=(0,0,0), epsilon=epsilon) for data_point in data_points])

    all_fp = []
    for i in range(0,len(masked_images),16):
        _, final_predictions_batch = run_joint_model(joint_model,(masked_images[i:i+16].to(device)))
        final_predictions_batch = torch.nn.Sigmoid()(final_predictions_batch.detach().cpu().T).numpy()
        all_fp.append(final_predictions_batch)
    final_predictions_batch = np.concatenate(all_fp)
    avg_diff = []

    for i in range(len(data_points)):
        relevant_indices = initial_predictions[data_points[i],parts_to_attribute[str(main_part)]]>0.75
        if True in relevant_indices:
            avg_diff.append(np.mean(np.abs(final_predictions_batch[i,parts_to_attribute[str(main_part)]][relevant_indices]-initial_predictions[data_points[i],parts_to_attribute[str(main_part)]][relevant_indices])))
    diff_by_epsilon[epsilon] = (np.mean(avg_diff),np.std(avg_diff))

    plt.imshow(numpy_to_pil(masked_images[1].numpy()))
    numpy_to_pil(masked_images[1].numpy()).save("../../results/cub/mask_creation/example_{}_{}.png".format(epsilon,seed))
# -

numpy_to_pil(masked_images[1].numpy()).save("../../results/cub/mask_creation/example_{}_{}.png".format(epsilon,seed))

results = {
    'diff_by_epsilon': diff_by_epsilon, 
    'part': main_part,
    'parameters': {
        'seed': seed, 
        'epsilon': epsilon, 
        'parts': part_list, 
        'images_per_mask': test_data_num, 
        'dataset': 'CUB', 
    }, 
}

for i in diff_by_epsilon:
    diff_by_epsilon[i] = (float(diff_by_epsilon[i][0]),float(diff_by_epsilon[i][1]))

save_name = "mask_{}.json".format(seed)

json.dump(results,open("../../results/cub/mask_creation/{}".format(save_name),"w"))
