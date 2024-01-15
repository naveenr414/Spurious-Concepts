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
import resource 

from src.images import *
from src.util import *
from src.models import *
from src.plot import *

torch.cuda.set_per_process_memory_fraction(0.5)
resource.setrlimit(resource.RLIMIT_AS, (30 * 1024 * 1024 * 1024, -1))
torch.set_num_threads(1)

# +
is_jupyter = 'ipykernel' in sys.modules
if is_jupyter:
    encoder_model='inceptionv3'
    seed = 42
    retrain_epochs = 0
    pruning_technique = 'weight'
    prune_rate = 0.25
    dataset_name = "coco"
else:
    parser = argparse.ArgumentParser(description="Synthetic Dataset Experiments")


    parser.add_argument('--encoder_model', type=str, default='inceptionv3', help='Encoder model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--retrain_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--pruning_technique', type=str, default='weight', help='"layer" or "weight" pruning')
    parser.add_argument('--prune_rate', type=float, default=0.25, help='Rate of pruning')

    args = parser.parse_args()
    encoder_model = args.encoder_model 
    seed = args.seed 
    retrain_epochs = args.retrain_epochs 
    pruning_technique = args.pruning_technique 
    prune_rate = args.prune_rate
    dataset_name = "coco"

parameters = {
    'seed': seed, 
    'encoder_model': encoder_model ,
    'retrain_epochs': retrain_epochs,
    'pruning_technique': pruning_technique,  
    'num_attributes': 10,
    'debugging': False,
    'prune_rate': prune_rate, 
    'dataset_name': dataset_name,
}
print(parameters)

# -

np.random.seed(seed)
torch.manual_seed(seed)

train_loader, val_loader, test_loader, train_pkl, val_pkl, test_pkl = get_data(1,encoder_model=encoder_model,dataset_name=dataset_name)

test_images, test_y, test_c = unroll_data(test_loader)

rand_name = secrets.token_hex(4)
results_file = "../../results/coco_pruning/{}.json".format(rand_name)
delete_same_dict(parameters,"../../results/coco_pruning")

model_parameters = {
    'debugging': False, 
    'encoder_model': encoder_model, 
    'seed': seed, 
    'dataset': 'coco', 
    'epochs': 25,
    'lr': 0.005, 
    "attr_loss_weight": 0.1, 
    'scheduler': 'none',
    'train_variation': 'none'
}
print(model_parameters)

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

torch.save(joint_model,open("../../models/pruned/coco/{}.pt".format(rand_name),"wb"))

joint_model = None 

torch.cuda.empty_cache()

command_to_run = "python train_cbm.py -dataset coco -epochs {} --load_model pruned/coco/{}.pt -num_attributes 10 --encoder_model {} -num_classes 2 -seed {} -lr 0.005".format(retrain_epochs,rand_name,encoder_model,seed)

command_to_run

subprocess.run("cd ../../ConceptBottleneck && {}".format(command_to_run),shell=True)

os.remove("../../models/pruned/coco/{}.pt".format(rand_name))

# +
joint_location = "../../models/pruned/coco/{}/joint/best_model_{}.pth".format(rand_name,seed)
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

train_locations = json.load(open(dataset_directory+"/coco/preprocessed/instances_train2014.json"))['annotations']
val_locations = json.load(open(dataset_directory+"/coco/preprocessed/instances_val2014.json"))['annotations']

concepts = pickle.load(open(dataset_directory+"/coco/preprocessed/concepts.pkl","rb"))

locations_by_image = {}
image_ids = set([i['id'] for i in train_pkl + val_pkl + test_pkl])
id_to_idx = {}

for i in train_locations + val_locations:
    if i ['image_id'] in image_ids and i['category_id'] in concepts:
        if i['image_id'] not in locations_by_image:
            locations_by_image[i['image_id']] = [[] for i in range(len(concepts))]
        locations_by_image[i['image_id']][concepts.index(i['category_id'])].append(i['bbox'])

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

results_by_part_mask = {}
test_data_num = 100
valid_pairs = [(i,j) for i in range(len(concepts)) for j in range(len(concepts)) if len(
    [k for k in range(len(test_pkl)) if test_c[k][i] == 1 and test_c[k][j] == 1]) > test_data_num]

for (main_part,mask_part) in valid_pairs:
    print("On main part {}".format(main_part))
    if concepts[main_part] not in results_by_part_mask:
        results_by_part_mask[concepts[main_part]] = {}

    test_images, test_y, test_c = unroll_data(test_loader)
    valid_data_points = [k for k in range(len(test_pkl)) if test_c[k][main_part] == 1 and test_c[k][mask_part] == 1]
    data_points = random.sample(valid_data_points,test_data_num)
    masked_dataset = [mask_bbox(test_images[idx],[get_new_x_y(locations_by_image[test_pkl[idx]['id']][mask_part][k],idx,test_pkl) for k in range(len(locations_by_image[test_pkl[idx]['id']][mask_part]))]) for idx in data_points]
    masked_dataset = torch.stack(masked_dataset)

    final_predictions = None 
    with torch.no_grad():
        _, final_predictions_batch = run_joint_model(joint_model,masked_dataset.to(device))
        final_predictions_batch = torch.nn.Sigmoid()(final_predictions_batch.detach().cpu().T)
        final_predictions = final_predictions_batch.numpy()     
    avg_diff = np.mean(np.abs(initial_predictions[data_points] - final_predictions)[:,main_part])
    std_diff = np.std(np.abs(initial_predictions[data_points] - final_predictions)[:,main_part])

    results_by_part_mask[concepts[main_part]][concepts[mask_part]] = (float(avg_diff),float(std_diff))

shutil.rmtree("../../models/pruned/coco/{}".format(rand_name))

final_data = {
    'train_accuracy': float(train_acc), 
    'val_accuracy': float(val_acc), 
    'test_accuracy': float(test_acc), 
    'results_by_part_mask': results_by_part_mask,  
    'parameters': parameters,  
}

json.dump(final_data,open(results_file,"w"))
