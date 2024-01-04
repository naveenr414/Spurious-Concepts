# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: concepts
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
import json
import argparse 
import secrets
import subprocess
import shutil 
from torch.nn.utils import prune

from src.images import *
from src.util import *
from src.models import *
from src.plot import *

# +
is_jupyter = 'ipykernel' in sys.modules
if is_jupyter:
    num_objects = 2
    encoder_model='inceptionv3'
    seed = 42
    retrain_epochs = 0
    pruning_technique = 'weight'
    expand_dim_encoder = 0
    num_middle_encoder = 0
    prune_rate = 0.25
    dataset_name = "CUB"
else:
    parser = argparse.ArgumentParser(description="Synthetic Dataset Experiments")


    parser.add_argument('--num_objects', type=int, default=1, help='Number of objects')
    parser.add_argument('--encoder_model', type=str, default='small3', help='Encoder model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--retrain_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--expand_dim_encoder', type=int, default=0, help='For MLPs, size of the middle layer')
    parser.add_argument('--num_middle_encoder', type=int, default=0, help='For MLPs, number of middle layers')
    parser.add_argument('--pruning_technique', type=str, default='weight', help='"layer" or "weight" pruning')
    parser.add_argument('--prune_rate', type=float, default=0.25, help='Rate of pruning')
    parser.add_argument('--dataset_name', type=str, default='synthetic_object/synthetic_1', help='Which dataset to use')

    args = parser.parse_args()
    num_objects = args.num_objects
    encoder_model = args.encoder_model 
    seed = args.seed 
    retrain_epochs = args.retrain_epochs 
    pruning_technique = args.pruning_technique 
    prune_rate = args.prune_rate
    dataset_name = args.dataset_name

parameters = {
    'seed': seed, 
    'encoder_model': encoder_model ,
    'retrain_epochs': retrain_epochs,
    'pruning_technique': pruning_technique,  
    'num_attributes': 112,
    'debugging': False,
    'prune_rate': prune_rate, 
    'dataset_name': dataset_name,
}
print(parameters)

# -

np.random.seed(seed)
torch.manual_seed(seed)

train_loader, val_loader, test_loader, train_pkl, val_pkl, test_pkl = get_data(num_objects,encoder_model=encoder_model,dataset_name=dataset_name)

test_images, test_y, test_c = unroll_data(test_loader)

rand_name = secrets.token_hex(4)
results_file = "../../results/cub_pruning/{}.json".format(rand_name)
delete_same_dict("../../results/cub_pruning",parameters)

# +
model_parameters = {
    'debugging': False, 
    'epochs': 100,
    'encoder_model': encoder_model, 
    'seed': seed, 
    'dataset': 'CUB', 
    'epochs': 100,
    'lr': 0.005

}
# -

device = 'cuda' if torch.cuda.is_available() else 'cpu'

joint_model = get_synthetic_model(dataset_name,model_parameters)

joint_model = joint_model.to(device)


# ## Prune Model

def find_conv2d_modules(model):
    conv2d_modules = []

    def find_conv2d_recursively(module):
        for child_module in module.children():
            if isinstance(child_module, nn.Conv2d):
                conv2d_modules.append(child_module)
            else:
                find_conv2d_recursively(child_module)

    find_conv2d_recursively(model)
    return conv2d_modules


if pruning_technique == "layer":
    for conv_number in [4,5,6,7]:
        if len(joint_model.first_model.conv_layers) >= conv_number: 
            layer_to_prune = joint_model.first_model.conv_layers[conv_number-1]
            weight = layer_to_prune.weight.data.abs().clone()
            importance = weight.sum((1, 2, 3))  # Calculate importance of filters
            num_filters = layer_to_prune.weight.size(0)

            # Compute the number of filters to prune
            num_prune = int(num_filters * prune_rate)
            _, indices = importance.sort(descending=True)
            indices_to_prune = indices[-num_prune:]

            # Create a mask to prune filters
            mask = torch.ones(num_filters)
            mask[indices_to_prune] = 0
            if mask is not None:
                mask = mask.to(layer_to_prune.weight.device)
                layer_to_prune.weight.data *= mask.view(-1, 1, 1, 1)    
elif pruning_technique == "weight":
    for conv_2d in find_conv2d_modules(joint_model.first_model):
        torch.nn.utils.prune.l1_unstructured(conv_2d, name="weight", amount=prune_rate) 
    for layer in joint_model.first_model.all_fc:
        layer = layer.fc 
        prune.l1_unstructured(layer, name="weight", amount=prune_rate)
else:
    raise Exception("Pruning {} not found".format(pruning_technique))

# ## Retraining

torch.save(joint_model,open("../../models/pruned/cub/{}.pt".format(rand_name),"wb"))

joint_model = None 

torch.cuda.empty_cache()

command_to_run = "python train_cbm.py -dataset CUB -epochs {} --load_model pruned/cub/{}.pt -num_attributes 112 --encoder_model {} -num_classes 200 -seed {}".format(retrain_epochs,rand_name,encoder_model,seed)

command_to_run

subprocess.run("cd ../../ConceptBottleneck && {}".format(command_to_run),shell=True)

os.remove("../../models/pruned/cub/{}.pt".format(rand_name))

# +
joint_location = "../../models/pruned/cub/{}/joint/best_model_{}.pth".format(rand_name,seed)
joint_model = torch.load(joint_location,map_location='cpu')

if 'encoder_model' in parameters and 'mlp' in parameters['encoder_model']:
    joint_model.encoder_model = True

r = joint_model.eval()
# -

joint_model = joint_model.to(device)

torch.cuda.empty_cache()

# ## Compute Activation + Accuracy

train_acc =  get_accuracy(joint_model,run_joint_model,train_loader)
val_acc = get_accuracy(joint_model,run_joint_model,val_loader)
test_acc =get_accuracy(joint_model,run_joint_model,test_loader)

train_acc, val_acc, test_acc  

dataset_directory = "../../../../datasets"

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

results_by_part_mask = {}

epsilon = 0.3
test_data_num = 100

for main_part in valid_parts[:1]:
    print("Main part is {}".format(main_part))
    results_by_part_mask[part_list[main_part]] = {}
    for mask_part in valid_parts:
        main_attributes = parts_to_attribute[str(main_part)]
        mask_attributes = parts_to_attribute[str(mask_part)]
        test_images, test_y, test_c = unroll_data(test_loader)

        valid_data_points = [i for i in range(len(test_pkl)) if main_part in locations_by_image_id[test_pkl[i]['id']] and mask_part in locations_by_image_id[test_pkl[i]['id']]]
        data_points = random.sample(valid_data_points,test_data_num)
        other_part_locations = [[get_part_location(data_point,new_part, locations_by_image_id, test_pkl) for new_part in valid_parts if new_part!=mask_part and new_part in locations_by_image_id[
            test_pkl[data_point]['id']]] for data_point in data_points]

        masked_dataset = [mask_image_closest(test_images[data_points[idx]],get_part_location(data_points[idx],mask_part, locations_by_image_id, test_pkl),other_part_locations[idx],epsilon=epsilon) for idx in range(len(data_points))]
        masked_dataset = torch.stack(masked_dataset)    

        final_predictions = None 
        with torch.no_grad():
            _, final_predictions_batch = run_joint_model(joint_model,masked_dataset.to(device))
            final_predictions_batch = torch.nn.Sigmoid()(final_predictions_batch.detach().cpu().T)
            final_predictions = final_predictions_batch.numpy()     
        avg_diff = np.mean(np.abs(initial_predictions[data_points] - final_predictions)[:,main_attributes])
        std_diff = np.std(np.abs(initial_predictions[data_points] - final_predictions)[:,main_attributes])

        results_by_part_mask[part_list[main_part]][part_list[mask_part]] = (avg_diff,std_diff)

for i in results_by_part_mask:
    for j in results_by_part_mask[i]:
        results_by_part_mask[i][j] = (float(results_by_part_mask[i][j][0]),float(results_by_part_mask[i][j][1]))

shutil.rmtree("../../models/pruned/cub/{}".format(rand_name))

final_data = {
    'train_accuracy': float(train_acc), 
    'val_accuracy': float(val_acc), 
    'test_accuracy': float(test_acc), 
    'results_by_part_mask': results_by_part_mask,  
    'parameters': parameters,  
}

json.dump(final_data,open(results_file,"w"))


