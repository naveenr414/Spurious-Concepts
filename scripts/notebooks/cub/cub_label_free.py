# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: cem
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

import torch
import sys
import matplotlib.pyplot as plt
from PIL import Image
import json
import argparse
import logging 
import gc 
from locality.cbm_variants.label_free import cbm

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
    seed = 44
    dataset_name = "CUB"
    train_variation = "loss"
else:
    parser = argparse.ArgumentParser(description="Synthetic Dataset Experiments")

    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--train_variation', type=str, default='none', help='Which train variation to analyze')

    args = parser.parse_args()
    encoder_model = "inceptionv3" 
    seed = args.seed 
    dataset_name = "CUB"
    train_variation = args.train_variation

parameters = {
    'dataset': dataset_name,
    'seed': seed, 
    'encoder_model': encoder_model ,
    'debugging': False,
    'epochs': 100,
    'lr': 0.005,
    'train_variation': train_variation,
}

# -

train_loader, val_loader, test_loader, train_pkl, val_pkl, test_pkl = get_data(1,encoder_model=encoder_model,dataset_name=dataset_name,get_label_free=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if is_jupyter:
    load_dir = "../../../locality/cbm_variants/label_free/cub_lf_cbm"
else:
    load_dir = "locality/cbm_variants/label_free/cub_lf_cbm"
with open(os.path.join(load_dir, "concepts.txt"), "r") as f:
    concepts = f.read().split("\n")

model = cbm.load_cbm(load_dir, device)

# ## Plot the Dataset

logging.info("Plotting Dataset")

if is_jupyter:
    dataset_directory = "../../../datasets"
else:
    dataset_directory = "datasets"

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
# -

label_free_attributes = [[c for c in concepts if p in c] for p in part_list]

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
        _, concept_act = model(x.cuda())
        initial_predictions.append(torch.nn.Sigmoid()(concept_act.detach().cpu()).numpy())
    initial_predictions = np.row_stack(initial_predictions)
    

torch.cuda.empty_cache()

valid_parts = [int(i) for i in parts_to_attribute if '^' not in i and parts_to_attribute[i] != [] and label_free_attributes[int(i)] != []]

results_by_part_mask = {}

epsilon = 0.3
test_data_num = 100

main_part = valid_parts[0]
mask_part = valid_parts[0]
results_by_part_mask[part_list[main_part]] = {}

test_images, test_y, test_c = unroll_data(test_loader)

# +
main_attributes = parts_to_attribute[str(main_part)]
mask_attributes = parts_to_attribute[str(mask_part)]
test_images, test_y, test_c = None, None, None 
gc.collect() 
test_images, test_y, test_c = unroll_data(test_loader)

valid_data_points = [i for i in range(len(test_pkl)) if main_part in locations_by_image_id[test_pkl[i]['id']] and mask_part in locations_by_image_id[test_pkl[i]['id']]]
data_points = random.sample(valid_data_points,test_data_num)
other_part_locations = [[get_part_location_center_crop(data_point,new_part, locations_by_image_id, test_pkl) for new_part in valid_parts if new_part!=mask_part and new_part in locations_by_image_id[
    test_pkl[data_point]['id']]] for data_point in data_points]

# -

for main_part in valid_parts:
    print("Main part is {}".format(main_part))
    results_by_part_mask[part_list[main_part]] = {}
    for mask_part in valid_parts:
        main_attributes = parts_to_attribute[str(main_part)]
        mask_attributes = parts_to_attribute[str(mask_part)]
        test_images, test_y, test_c = None, None, None 
        gc.collect() 
        test_images, test_y, test_c = unroll_data(test_loader)

        valid_data_points = [i for i in range(len(test_pkl)) if main_part in locations_by_image_id[test_pkl[i]['id']] and mask_part in locations_by_image_id[test_pkl[i]['id']]]
        data_points = random.sample(valid_data_points,test_data_num)
        other_part_locations = [[get_part_location_center_crop(data_point,new_part, locations_by_image_id, test_pkl) for new_part in valid_parts if new_part!=mask_part and new_part in locations_by_image_id[
            test_pkl[data_point]['id']]] for data_point in data_points]

        masked_dataset = [mask_image_closest(test_images[data_points[idx]],get_part_location_center_crop(data_points[idx],mask_part, locations_by_image_id, test_pkl),other_part_locations[idx],epsilon=epsilon,color=torch.mean(test_images,dim=(0,2,3)).numpy().astype(np.float64)*np.array([0.229, 0.224, 0.225])+np.array([0.485, 0.456, 0.406]),width=224,height=224,mean=np.array([0.485, 0.456, 0.406]),std=np.array([0.229, 0.224, 0.225])) for idx in range(len(data_points))]
        masked_dataset = torch.stack(masked_dataset)    

        with torch.no_grad():
            final_predictions = [] 

            for i in range(0,len(masked_dataset),64):
                _, concept_act = model(masked_dataset[i:i+64].cuda())
                final_predictions.append(torch.nn.Sigmoid()(concept_act.detach().cpu()).numpy())
            final_predictions = np.row_stack(final_predictions)

        avg_diff = np.mean(np.abs(initial_predictions[data_points] - final_predictions)[:,main_attributes])
        std_diff = np.std(np.abs(initial_predictions[data_points] - final_predictions)[:,main_attributes])

        results_by_part_mask[part_list[main_part]][part_list[mask_part]] = (avg_diff,std_diff)

results = {
    'part_mask': results_by_part_mask, 
    'parameters': {
        'seed': seed, 
        'epsilon': epsilon, 
        'parts': part_list, 
        'images_per_mask': test_data_num, 
        'dataset': 'CUB', 
        'train_variation': 'label-free', 
    }, 
}

for i in results['part_mask']:
    for j in results['part_mask'][i]:
        results['part_mask'][i][j] = (float(results['part_mask'][i][j][0]),float(results['part_mask'][i][j][1]))

# +
# save_name = "mask_epsilon_{}.json".format(seed)
# if train_variation != 'none':
#     save_name = "mask_epsilon_{}_{}.json".format(train_variation,seed)

save_name = "mask_epsilon_mean_color_{}.json".format(seed)
# -

if is_jupyter:
    json.dump(results,open("../../../results/cub/{}".format(save_name),"w"))
else:
    json.dump(results,open("results/cub/{}".format(save_name),"w"))


