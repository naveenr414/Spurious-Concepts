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
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import logging 
import sys

from src.cbm_variants.ConceptBottleneck.CUB.dataset import load_data

from src.images import *
from src.util import *
from src.models import *
from src.plot import *

# ## Set up dataset + model

logging.basicConfig(level=logging.INFO)
logging.info("Setting up dataset")

# +
is_jupyter = 'ipykernel' in sys.modules
if is_jupyter:
    encoder_model='small7'
    seed = 44
    dataset_name = "synthetic_object/synthetic_2"
    train_variation = "none"
    model_type = "cem"
else:
    parser = argparse.ArgumentParser(description="Synthetic Dataset Experiments")

    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--train_variation', type=str, default='none', help='Which train variation to analyze')
    parser.add_argument('--model_type', type=str, default='joint', help='Which train variation to analyze')

    args = parser.parse_args()
    encoder_model = "inceptionv3" 
    seed = args.seed 
    dataset_name = "synthetic_object/synthetic_2"
    train_variation = args.train_variation
    model_type = args.model_type

parameters = {
    'dataset': dataset_name,
    'seed': seed, 
    'encoder_model': encoder_model ,
    'debugging': False,
    'epochs': 50,
    'lr': 0.05,
    'train_variation': train_variation,
    'model_type': model_type,
    'weight_decay': 0.004, 
    'scale_lr': 5
}

# -

train_loader, val_loader, test_loader, train_pkl, val_pkl, test_pkl = get_data(1,encoder_model=encoder_model,dataset_name=dataset_name)

reference_pkl = [
    {'id': 0, 'img_path': 'synthetic_object/synthetic_2/reference_images/1_0_0_0.png','class_label': 0, 'attribute_label': [1,0,0,0]},
    {'id': 1, 'img_path': 'synthetic_object/synthetic_2/reference_images/0_1_0_0.png','class_label': 1, 'attribute_label': [0,1,0,0]},
    {'id': 2, 'img_path': 'synthetic_object/synthetic_2/reference_images/0_0_1_0.png','class_label': 0, 'attribute_label': [0,0,1,0]},
    {'id': 3, 'img_path': 'synthetic_object/synthetic_2/reference_images/0_0_0_1.png','class_label': 0, 'attribute_label': [0,0,0,1]}
]

pickle.dump(reference_pkl,open("../../../../datasets/synthetic_object/synthetic_2/preprocessed/reference.pkl","wb"))

reference_image_path = "../../../../datasets/synthetic_object/synthetic_2/preprocessed/reference.pkl"
resize =  "inceptionv3" in encoder_model
get_label_free=False 
reference_loader = load_data([reference_image_path], True, no_img=False, batch_size=64, image_dir="reference_images/", n_class_attr=2, path_transform=lambda path: "../../../../datasets/"+path,resize=resize,get_label_free=get_label_free)


test_images, test_y, test_c = unroll_data(test_loader)

reference_images, references_y, references_c = unroll_data(reference_loader)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if model_type != 'joint':
    joint_model = get_synthetic_model(dataset_name,{'model_type': model_type, 'dataset': 'synthetic_2', 'seed': seed})
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

dataset_directory = "../../../../datasets"

train_pkl[0]

img_path = dataset_directory+'/'+train_pkl[0]['img_path']
image = Image.open(img_path)
plt.imshow(image)

# ## Accuracy

train_acc =  get_accuracy(joint_model,run_model_function,train_loader)
val_acc = get_accuracy(joint_model,run_model_function,val_loader)
test_acc = get_accuracy(joint_model,run_model_function,test_loader)

train_acc, val_acc, test_acc

# +
with torch.no_grad():

    for data_point in reference_loader:
        x,y,c = data_point 
        _, reference_predictions = run_model_function(joint_model,x.to(device))
        reference_predictions = torch.nn.Sigmoid()(reference_predictions.detach().cpu().T)
    
relevant_mask = []
for concept in range(2):
    for data_point in [[1,0,1,0],[1,0,0,1],[0,1,1,0],[0,1,0,1]]:
        if concept == 0:
            initial_idx = data_point[:2].index(1)
        else:
            initial_idx = data_point[2:].index(1)+2
        data_point[initial_idx] = 0
        new_prediction = reference_predictions[data_point.index(1)][initial_idx]
        relevant_mask.append(1-new_prediction)

irrelevant_mask = []
for concept in range(2):
    for data_point in [[1,0,1,0],[1,0,0,1],[0,1,1,0],[0,1,0,1]]:
        if concept == 0:
            initial_idx = data_point[:2].index(1)
            other_idx = data_point[2:].index(1)+2
        else:
            initial_idx = data_point[2:].index(1)+2
            other_idx = data_point[:2].index(1)
        data_point[other_idx] = 0
        new_prediction = reference_predictions[data_point.index(1)][initial_idx]
        irrelevant_mask.append(1-new_prediction)
# -

results = {
    'relevant_mask': relevant_mask,
    'irrelevant_mask': irrelevant_mask,
    'parameters': {
        'seed': seed, 
        'dataset': 'synthetic_object/synthetic_2', 
        'train_variation': train_variation, 
        'model_type': model_type, 
    }, 
}
