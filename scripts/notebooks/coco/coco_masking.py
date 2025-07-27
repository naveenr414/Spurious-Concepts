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

import torch
import sys
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import json
import argparse
import logging 
from torch.utils.data import Subset

from locality.images import *
from locality.util import *
from locality.models import *

# ## Set up dataset + model

logging.basicConfig(level=logging.INFO)
logging.info("Setting up dataset")

# +
is_jupyter = 'ipykernel' in sys.modules
if is_jupyter:
    encoder_model='inceptionv3'
    seed = 42
    dataset_name = "coco"
    train_variation = "none"
    model_type = "cem"
    dataset_directory = "../../../datasets/"
else:
    parser = argparse.ArgumentParser(description="Synthetic Dataset Experiments")

    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--train_variation', type=str, default='none', help='Which train variation to analyze')
    parser.add_argument('--model_type', type=str, default='joint', help='Which train variation to analyze')

    args = parser.parse_args()
    encoder_model = "inceptionv3" 
    seed = args.seed 
    dataset_name = "coco"
    train_variation = args.train_variation
    model_type = args.model_type
    dataset_directory = "datasets/"

parameters = {
    'dataset': dataset_name,
    'seed': seed, 
    'encoder_model': encoder_model ,
    'debugging': False,
    'epochs': 25,
    'lr': 0.005,
    'train_variation': train_variation,
    'model_type': model_type,
}

# -

train_loader, val_loader, test_loader, train_pkl, val_pkl, test_pkl = get_data(1,encoder_model=encoder_model,dataset_name=dataset_name)

test_images, test_y, test_c = unroll_data(test_loader)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if model_type != 'joint':
    joint_model = get_synthetic_model(dataset_name,{'model_type': model_type, 'dataset': 'coco', 'seed': seed})
else:
    joint_model = get_synthetic_model(dataset_name,parameters)
joint_model = joint_model.to(device)


run_model_function = run_joint_model
if model_type == "cem":
    run_model_function = run_cem_model
elif model_type == "probcbm":
    run_model_function = run_probcbm_model

# ## Plot the Dataset

logging.info("Plotting Dataset")

# +
concepts = pickle.load(open(dataset_directory+"/coco/preprocessed/concepts.pkl","rb"))
concept_names = json.load(open(dataset_directory+"/coco/preprocessed/instances_train2014.json"))['categories']
concept_names_from_id = {}

for i in concept_names:
    concept_names_from_id[i['id']] = i['name']
relevant_concepts = [concept_names_from_id[i] for i in concepts]
# -

img_path = dataset_directory+'/'+train_pkl[0]['img_path']
image = Image.open(img_path)
plt.imshow(image)
[relevant_concepts[i] for i,val in enumerate(train_pkl[0]['attribute_label']) if val == 1]

# ## Accuracy

train_acc =  get_accuracy(joint_model,run_model_function,train_loader)
val_acc = get_accuracy(joint_model,run_model_function,val_loader)
test_acc = get_accuracy(joint_model,run_model_function,test_loader)

train_acc, val_acc, test_acc

if model_type == 'joint':
    concept_acc = get_concept_accuracy_by_concept(joint_model,run_model_function,test_loader)
    locality_intervention = 1-torch.mean(concept_acc).detach().numpy()
    json.dump({'locality_intervention': locality_intervention},open("../../results/coco/locality_intervention_{}.json".format(seed),"w"))

torch.cuda.empty_cache()

# ## Using Part Annotations

logging.info("Using part annotations")

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

logging.info("Impact of Epsilon")

with torch.no_grad():
    initial_predictions = [] 

    for data_point in test_loader:
        x,y,c = data_point 
        _, initial_predictions_batch = run_model_function(joint_model,x.to(device))
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
    valid_data_points = [k for k in range(len(test_pkl)) if test_c[k][main_part] == 1 and test_c[k][mask_part] == 1]
    data_points = random.sample(valid_data_points,test_data_num)
    subset_loader = torch.utils.data.DataLoader(
        Subset(test_loader.dataset, data_points),
        batch_size=len(data_points),  # Load all at once for efficiency
        shuffle=False,
        num_workers=test_loader.num_workers,
        pin_memory=test_loader.pin_memory
    )

    test_images, test_y, _ = unroll_data(subset_loader)
    masked_dataset = [mask_bbox(test_images[i],[get_new_x_y(locations_by_image[test_pkl[idx]['id']][mask_part][k],idx,test_pkl) for k in range(len(locations_by_image[test_pkl[idx]['id']][mask_part]))]) for i,idx in enumerate(data_points)]
    masked_dataset = torch.stack(masked_dataset)

    final_predictions = None 
    with torch.no_grad():
        _, final_predictions_batch = run_model_function(joint_model,masked_dataset.to(device))
        final_predictions_batch = torch.nn.Sigmoid()(final_predictions_batch.detach().cpu().T)
        final_predictions = final_predictions_batch.numpy()     
    avg_diff = np.mean(np.abs(initial_predictions[data_points] - final_predictions)[:,main_part])
    std_diff = np.std(np.abs(initial_predictions[data_points] - final_predictions)[:,main_part])

    results_by_part_mask[concepts[main_part]][concepts[mask_part]] = (float(avg_diff),float(std_diff))

results = {
    'part_mask': results_by_part_mask, 
    'parameters': {
        'seed': seed, 
        'concepts': concepts, 
        'images_per_mask': test_data_num, 
        'dataset': 'coco', 
        'train_variation': train_variation, 
        'model_type': model_type, 
    }, 
    'train_acc': train_acc,
    'val_acc': val_acc,
    'test_acc': test_acc,
}

if model_type == 'joint':
    json.dump(results,open("../../results/coco/mask_{}_{}.json".format(train_variation,seed),"w"))
else:
    json.dump(results,open("../../results/coco/mask_{}_{}.json".format(model_type,seed),"w"))
