# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: concepts
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

import sys
import torch
import json
import argparse 
import subprocess

from locality.images import *
from locality.util import *
from locality.models import *

# +
is_jupyter = 'ipykernel' in sys.modules
if is_jupyter:
    encoder_model='inceptionv3'
    seed = 43
    dataset_name = 'CUB'
    num_concept_combinations = 200
else:
    parser = argparse.ArgumentParser(description="Synthetic Dataset Experiments")


    parser.add_argument('--encoder_model', type=str, default='inceptionv3', help='Encoder model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dataset_name', type=str, default='CUB', help='Number of concept combinations')
    parser.add_argument('--num_concept_combinations', type=int, default=100, help='Random seed')

    args = parser.parse_args()
    encoder_model = args.encoder_model 
    seed = args.seed 
    dataset_name = args.dataset_name 
    num_concept_combinations = args.num_concept_combinations

parameters = {
    'seed': seed, 
    'encoder_model': encoder_model ,
    'debugging': False,
    'dataset_name': dataset_name,
    'num_concept_combinations': num_concept_combinations,
}
print(parameters)
# -

np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

train_loader, val_loader, test_loader, train_pkl, val_pkl, test_pkl = get_data(1,encoder_model=encoder_model,dataset_name=dataset_name)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def generate_random_combinations(K):
    all_combos = list(set([str(i['attribute_label']) for i in train_pkl]))
    random.seed(seed)    
    # Generate all possible combinations
    random.shuffle(all_combos)
    all_combos = [eval(i) for i in all_combos]

    return all_combos[:K]


random_combinations = generate_random_combinations(num_concept_combinations)
formatted_combinations = []
for r in random_combinations:
    formatted_combinations.append(str(int("".join([str(i) for i in r]),2)))

if dataset_name == "CUB":
    epochs = 100
    command_to_run = "python train_cbm.py -dataset CUB --encoder_model inceptionv3 --pretrained -epochs {} -num_attributes 112 -num_classes 200 -seed {} --attr_loss_weight 0.01 --optimizer adam --scheduler_step 100 -lr 0.005 --concept_restriction {}".format(epochs,seed," ".join(formatted_combinations))
elif dataset_name == "coco":
    epochs = 25
    command_to_run = "python train_cbm.py -dataset coco --encoder_model inceptionv3 --pretrained -epochs {} -num_attributes 10 -num_classes 2 -seed {} --attr_loss_weight 0.1 --optimizer adam --scheduler_step 100 -lr 0.005 --concept_restriction {}".format(epochs,seed," ".join(formatted_combinations))

subprocess.run("cd locality/cbm_variants/ConceptBottleneck && {}".format(command_to_run),shell=True)


def get_most_recent_file(directory):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    if not files:
        return None

    most_recent_file = max(files, key=os.path.getmtime)
    
    return most_recent_file



most_recent_data = get_most_recent_file("../../../models/model_data/")
rand_name = most_recent_data.split("/")[-1].replace(".json","")
results_file = "../../../results/correlation/{}.json".format(rand_name)
delete_same_dict(parameters,"../../../results/correlation")

seed

rand_name = get_name_matching_parameters({'seed': seed, 'dataset': dataset_name})#, 'concept_restrictions': str([int(i) for i in formatted_combinations])})
temp = [json.load(open("../../../models/model_data/{}.json".format(i))) for i in rand_name]
rand_name = [rand_name[i] for i in range(len(rand_name)) if 'concept_restriction' in temp[i] and len(temp[i]['concept_restriction']) == num_concept_combinations]
rand_name = rand_name[-1]
results_file = "../../../results/correlation/{}.json".format(rand_name)


# +
joint_location = "../../../models/{}/{}/joint/best_model_{}.pth".format(dataset_name,rand_name,seed)
joint_model = torch.load(joint_location,map_location='cpu')

if 'encoder_model' in parameters and 'mlp' in parameters['encoder_model']:
    joint_model.encoder_model = True

r = joint_model.eval()
# -

joint_model = joint_model.to(device)

torch.cuda.empty_cache()

# ## Compute Accuracy

concept_acc = get_concept_accuracy_by_concept(joint_model,run_joint_model,test_loader)
locality_intervention = 1-torch.mean(concept_acc).detach().numpy()

final_data = {
    'parameters': parameters,  
    'locality_intervention': locality_intervention,
}

final_data

json.dump(final_data,open(results_file,"w"))
