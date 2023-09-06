# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: cem
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

import os
os.chdir('../')

import sys
sys.path.append('/home/njr61/rds/hpc-work/spurious-concepts/ConceptBottleneck')
sys.path.append('/home/njr61/rds/hpc-work/spurious-concepts')

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

from ConceptBottleneck.CUB.models import ModelXtoC, ModelOracleCtoY
from ConceptBottleneck.CUB.dataset import load_data

from src.images import *
from src.util import *
from src.models import *
from src.plot import *

# ## Set up dataset + model

logging.basicConfig(level=logging.INFO)
logging.info("Setting up dataset")

dataset = 'CUB'
noisy=False
weight_decay = 0.0004
encoder_model='small3'
optimizer = 'sgd'

is_exploration = False

if is_exploration:
    seed = 42 
else:
    parser = argparse.ArgumentParser(description="Your script description here")

    # Add command-line arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Parse the command-line arguments
    args = parser.parse_args()

    seed = args.seed


def get_log_folder(dataset,weight_decay,encoder_model,optimizer):
    if weight_decay == 0.0004 and encoder_model == 'inceptionv3':
        log_folder = f"results/{dataset}/joint"
    elif weight_decay == 0.0004:
        log_folder = f"results/{dataset}/joint_model_{encoder_model}"
    elif encoder_model == 'inceptionv3':
        log_folder = f"results/{dataset}/joint_wd_{weight_decay}"
    else:
        log_folder = f"results/{dataset}/joint_model_{encoder_model}_wd_{weight_decay}"
    if optimizer != 'sgd':
        log_folder += "_opt_{}".format(optimizer)
    
    log_folder += '/joint'
    
    return log_folder


dataset_name = "CUB"
data_dir = "../cem/cem/{}/preprocessed/".format(dataset_name)
data_dir

train_data_path = os.path.join(data_dir, 'train.pkl')
val_data_path = train_data_path.replace('train.pkl', 'val.pkl')

pretrained = True
freeze = False
use_aux = True
expand_dim = 0
three_class = False
use_attr = True
no_img = False
batch_size = 64
uncertain_labels = False
image_dir = 'images'
num_class_attr = 2
resampling = False

train_loader = load_data([train_data_path], use_attr, no_img, batch_size, uncertain_labels, image_dir=image_dir, 
                         n_class_attr=num_class_attr, resampling=resampling, path_transform=lambda path: "../cem/cem/"+path, is_training=False)
val_loader = load_data([val_data_path], use_attr, no_img=False, batch_size=64, image_dir=image_dir, n_class_attr=num_class_attr, path_transform=lambda path: "../cem/cem/"+path)

log_folder = get_log_folder(dataset_name,weight_decay,encoder_model,optimizer)
joint_location = "ConceptBottleneck/{}/best_model_{}.pth".format(log_folder,seed)
print(joint_location)
joint_model = torch.load(joint_location,map_location=torch.device('cpu'))
r = joint_model.eval()

# ## Plot the Dataset

logging.info("Plotting Dataset")

train_pkl = pickle.load(open(train_data_path,"rb"))
val_pkl = pickle.load(open(val_data_path,"rb"))


val_images, val_y, val_c = unroll_data(val_loader)

if is_exploration: 
    num_images_show = 5
    for i in range(num_images_show):
        img_path = '../cem/cem/'+train_pkl[i]['img_path']
        image = Image.open(img_path)
        image_array = np.array(image)
        plt.figure()
        plt.imshow(image_array)
        plt.axis('off') 

if is_exploration: 
    get_accuracy(joint_model,run_joint_model,train_loader), get_accuracy(joint_model,run_joint_model,val_loader)

if is_exploration: 
    get_concept_accuracy_by_concept(joint_model,run_joint_model,train_loader,sigmoid=True)

# ## Saliency Maps

logging.info("Saliency Maps")

attributes = open("../cem/cem/CUB/metadata/attributes.txt").read().split("\n")

attribute_num = 0
dataset_num = 1

if is_exploration:
    plot_gradcam(joint_model,run_joint_model,attribute_num,val_images,dataset_num,val_pkl)

if is_exploration:
    plot_integrated_gradients(joint_model,run_joint_model,attribute_num,val_images,dataset_num)

if is_exploration:
    plot_saliency(joint_model,run_joint_model,attribute_num,val_images,dataset_num)

# ## Using Part Annotations

logging.info("Using part annotations")

part_file = open("../cem/cem/CUB/metadata/parts/part_locs.txt").read().strip().split("\n")

part_list = open("../cem/cem/CUB/metadata/parts/parts.txt").read().strip().split("\n")
part_list = [' '.join(i.split(' ')[1:]) for i in part_list]

attribute_names = open("../cem/cem/CUB/metadata/attributes.txt").read().strip().split("\n")
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

locations_by_image = {}
for i in part_file:
    split_vals = i.split(' ')

    if split_vals[-1] == '1':
        image_id = int(split_vals[0])
        part_id = int(split_vals[1])-1 # 0 index 
        x = float(split_vals[2])
        y = float(split_vals[3])

        if image_id not in locations_by_image:
            locations_by_image[image_id] = {}
        locations_by_image[image_id][part_id] = (x,y)

valid_parts = [int(i) for i in parts_to_attribute if '^' not in i and parts_to_attribute[i] != []]

# #### Impact of Epsilon on Predictions

batch_size = 16
closest_results = {
    'mean': {},
    'std': {},
}

for part_num in valid_parts: 
    logging.info("On part {}".format(part_list[part_num]))
    closest_results['mean'][part_list[part_num]] = {}
    closest_results['std'][part_list[part_num]] = {}
    for epsilon in [10,20,30,40,50]:
        print("on epsilon {}".format(epsilon))

        # Get the corresponding data points
        corresponding_attributes = parts_to_attribute[str(part_num)]

        def run_for_datapoints(batch_points):
            new_images = []

            for i in batch_points:
                part_location = get_part_location(i,part_num, locations_by_image, val_pkl)
                image_id = val_pkl[i]['id']
                other_parts = [get_part_location(i,new_part, locations_by_image, val_pkl) for new_part in range(len(part_list)) if new_part!=part_num and new_part in locations_by_image[image_id]]
                new_images.append(mask_image_closest(deepcopy(val_images[i]),part_location,other_parts,epsilon=epsilon))
        
            new_images = torch.stack(new_images)
            
            _, initial_predictions_batch = run_joint_model(joint_model,val_images[batch_points])
            initial_predictions_batch = torch.nn.Sigmoid()(initial_predictions_batch.T)

            _, final_predictions_batch = run_joint_model(joint_model,new_images)
            final_predictions_batch = torch.nn.Sigmoid()(final_predictions_batch.T)

            diff = initial_predictions_batch[:,corresponding_attributes] - final_predictions_batch[:,corresponding_attributes]
            diff = diff.detach().numpy() 

            return diff 

        data_points = [i for i in range(100) if sum(val_c[i][corresponding_attributes])>0]
        data_points = [i for i in data_points if part_num in locations_by_image[val_pkl[i]['id']]]

        diff = batch_run(run_for_datapoints,data_points,16)
        diff = np.concatenate(diff)

        diff_max = np.amax(np.abs(diff),axis=1)
        std = np.std(diff_max)
        diff = np.sum(diff_max)/len(diff_max)
        closest_results['mean'][part_list[part_num]][epsilon] = float(diff)
        closest_results['std'][part_list[part_num]][epsilon] = float(std)
# -

closest_results 

json.dump(closest_results,open("results/cub/mask_closest_{}.json".format(seed),"w"))

logging.info("Dumped closest results")

logging.info("Random results")
random_results = {
    'mean': {},
    'std': {},
}

for i in valid_parts:
    random_results['mean'][part_list[i]] = {}
    random_results['std'][part_list[i]] = {}
# -

for epsilon in [10,20,30,40,50]:
    all_data_points = list(range(100))
    _, initial_predictions = run_joint_model(joint_model,val_images[all_data_points])
    initial_predictions = torch.nn.Sigmoid()(initial_predictions.T)
    for part_num in valid_parts:
        print("On {}".format(part_num))

        corresponding_attributes = parts_to_attribute[str(part_num)]
        corresponding_data_points = [i for i in range(100) if sum(val_c[i][corresponding_attributes])>0]
        corresponding_data_points = [i for i in corresponding_data_points if part_num in locations_by_image[val_pkl[i]['id']]]

        def get_diff(data_points):
            new_images = torch.stack([mask_image_location(
            deepcopy(val_images[i]),(random.randint(0,298),random.randint(0,298)),epsilon=epsilon) for i in data_points])
            _, final_predictions = run_joint_model(joint_model,new_images)
            final_predictions = torch.nn.Sigmoid()(final_predictions.T)

            return (initial_predictions[data_points][:,corresponding_attributes] - final_predictions[:,corresponding_attributes]).detach().numpy()

        diff = batch_run(get_diff,corresponding_data_points,16)
        diff = np.concatenate(diff,axis=0)

        logging.info("Diff shape {}".format(diff.shape))

        diff_max = np.amax(np.abs(diff),axis=1)
        std = np.std(diff_max)
        diff = np.sum(diff_max)/len(diff_max)
        random_results['mean'][part_list[part_num]][epsilon] = float(diff)
        random_results['std'][part_list[part_num]][epsilon] = float(std.item())


json.dump(random_results,open("results/cub/mask_random_{}.json".format(seed),"w"))

random_results

logging.info("Impact of Epsilon")

diff_results = {
    'mean': {},
    'std': {},
}

# +
batch_size = 16

for part_num in valid_parts: 
    logging.info("On part {}".format(part_list[part_num]))
    diff_results['mean'][part_list[part_num]] = {}
    diff_results['std'][part_list[part_num]] = {}
    for epsilon in [10,20,30,40,50]:
        # Get the corresponding data points
        corresponding_attributes = parts_to_attribute[str(part_num)]

        def run_for_datapoints(batch_points):
            new_images = torch.stack([mask_part(i,part_num,locations_by_image,val_pkl,val_images,epsilon=epsilon) for i in batch_points])
            
            _, initial_predictions_batch = run_joint_model(joint_model,val_images[batch_points])
            initial_predictions_batch = torch.nn.Sigmoid()(initial_predictions_batch.T)

            _, final_predictions_batch = run_joint_model(joint_model,new_images)
            final_predictions_batch = torch.nn.Sigmoid()(final_predictions_batch.T)

            diff = initial_predictions_batch[:,corresponding_attributes] - final_predictions_batch[:,corresponding_attributes]
            diff = diff.detach().numpy() 

            return diff 

        data_points = [i for i in range(100) if sum(val_c[i][corresponding_attributes])>0]
        data_points = [i for i in data_points if part_num in locations_by_image[val_pkl[i]['id']]]

        diff = batch_run(run_for_datapoints,data_points,16)
        diff = np.concatenate(diff)

        diff_max = np.amax(np.abs(diff),axis=1)
        std = np.std(diff_max)
        diff = np.sum(diff_max)/len(diff_max)
        diff_results['mean'][part_list[part_num]][epsilon] = float(diff)
        diff_results['std'][part_list[part_num]][epsilon] = float(std)
# -

json.dump(diff_results,open("results/cub/mask_epsilon_{}.json".format(seed),"w"))

logging.info("Dumped mask epsilon results")

diff_results

# +



# +


