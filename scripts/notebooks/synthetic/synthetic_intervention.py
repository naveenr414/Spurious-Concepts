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

import torch
import itertools
import json
import argparse 
import subprocess
import sys

from locality.images import *
from locality.util import *
from locality.models import *

# +
is_jupyter = 'ipykernel' in sys.modules
if is_jupyter:
    encoder_model='small7'
    seed = 42
    num_concept_combinations = 8

    num_objects = 4
else:
    parser = argparse.ArgumentParser(description="Synthetic Dataset Experiments")


    parser.add_argument('--encoder_model', type=str, default='small3', help='Encoder model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_concept_combinations', type=int, default=1, help='Number of concept combinations')
    parser.add_argument('--num_objects', type=int, default=4, help='Number of objects/which synthetic dataset')

    args = parser.parse_args()
    encoder_model = args.encoder_model 
    seed = args.seed 
    num_concept_combinations = args.num_concept_combinations 
    num_objects = args.num_objects

dataset_name = "synthetic_object/synthetic_{}".format(num_objects)

parameters = {
    'seed': seed, 
    'encoder_model': encoder_model ,
    'debugging': False,
    'dataset_name': dataset_name,
    'num_concept_combinations': num_concept_combinations
}
print(parameters)
# -

np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

train_loader, val_loader, test_loader, train_pkl, val_pkl, test_pkl = get_data(num_objects,encoder_model=encoder_model,dataset_name=dataset_name)

test_images, test_y, test_c = unroll_data(test_loader)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def generate_random_combinations(L, K):
    random.seed(seed)    
    # Generate all possible combinations
    all_combinations = list(itertools.product([0, 1], repeat=L))
    random.shuffle(all_combinations)

    return all_combinations[:K]


random_combinations = generate_random_combinations(num_objects,num_concept_combinations)
random_full_combinations = []
for c in random_combinations:
    random_full_combinations.append([])
    for d in c:
        random_full_combinations[-1].append(d)
        random_full_combinations[-1].append(1-d)
formatted_combinations = []
for r in random_full_combinations:
    formatted_combinations.append(str(int("".join([str(i) for i in r]),2)))

command_to_run = "python train_cbm.py -dataset {} -epochs {} -num_attributes {} --encoder_model {} -num_classes 2 -seed {} --concept_restriction {}".format(dataset_name,50,num_objects*2,encoder_model,seed," ".join(formatted_combinations))

subprocess.run("cd ../../ConceptBottleneck && {}".format(command_to_run),shell=True)


def get_most_recent_file(directory):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    if not files:
        return None

    most_recent_file = max(files, key=os.path.getmtime)
    
    return most_recent_file



most_recent_data = get_most_recent_file("../../models/model_data/")
rand_name = most_recent_data.split("/")[-1].replace(".json","")
results_file = "../../results/correlation/{}.json".format(rand_name)
delete_same_dict(parameters,"../../results/correlation")

# +
joint_location = "../../models/synthetic_object/synthetic_{}/{}/joint/best_model_{}.pth".format(num_objects,rand_name,seed)
joint_model = torch.load(joint_location,map_location='cpu')

if 'encoder_model' in parameters and 'mlp' in parameters['encoder_model']:
    joint_model.encoder_model = True

r = joint_model.eval()
# -

joint_model = joint_model.to(device)

torch.cuda.empty_cache()

# ## Compute Accuracy

train_acc =  get_accuracy(joint_model,run_joint_model,train_loader)
val_acc = get_accuracy(joint_model,run_joint_model,val_loader)
test_acc =get_accuracy(joint_model,run_joint_model,test_loader)

activation_values = []
run_model_function = run_joint_model
for concept_num in range(num_objects*2):
    val_for_concept = 0
    trials = 5

    for _ in range(trials):
        data_point = random.randint(0,len(test_images)-1)
        input_image = deepcopy(test_images[data_point:data_point+1])
        current_concept_val = test_c[data_point][concept_num]

        ret_image = get_maximal_activation(joint_model,run_model_function,concept_num,
                                        get_valid_image_function(concept_num,num_objects,epsilon=32),fixed_image=input_image,current_concept_val=current_concept_val).to(device)
        predicted_concept = torch.nn.Sigmoid()(run_model_function(joint_model,ret_image)[1].detach().cpu())[concept_num][0].detach().numpy()
        
        val_for_concept += abs(predicted_concept-current_concept_val.detach().numpy())/trials 
    activation_values.append(val_for_concept)


# +
in_distro = 0
correct_in_distro = 0 

out_distro = 0
correct_out_distro = 0

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with torch.no_grad():  # Use torch.no_grad() to disable gradient computation

    for data in test_loader:
        x, y, c = data
        y_pred, c_pred = run_joint_model(joint_model, x.to(device))
        c_pred = torch.stack([i.detach() for i in c_pred])
        c_pred = torch.nn.Sigmoid()(c_pred)

        c_pred = c_pred.numpy().T
        y_pred = logits_to_index(y_pred.detach())

        c = torch.stack([i.detach() for i in c]).numpy().T

        in_distribution = []

        for i in range(len(c)):
            binary_c = c[i]
            combo = str(int("".join([str(i) for i in binary_c]),2))

            if combo in formatted_combinations:
                in_distribution.append(True)
            else:
                in_distribution.append(False)
        
        in_distro += in_distribution.count(True) * len(c[0])
        out_distro += in_distribution.count(False) * len(c[0])

        in_distribution = np.array(in_distribution)

        correct_in_distro += np.sum(np.clip(np.round(c_pred[in_distribution]),0,1) == c[in_distribution]).item() 
        correct_out_distro += np.sum(np.clip(np.round(c_pred[~in_distribution]),0,1) == c[~in_distribution]).item() 



# +
concept_accuracies = []

# Try and flip each concept
for concept_num in range(num_objects*2):
# Set this concept_num to 1 (which sets the corresponding thing to 0)
    total_flipped = 0
    total_points = 0

    with torch.no_grad():  # Use torch.no_grad() to disable gradient computation

        for data in test_loader:
            x, y, c = data
            y_pred, c_pred = run_joint_model(joint_model, x.to(device))
            c_pred = torch.stack([i.detach() for i in c_pred]).numpy().T
            y_pred = logits_to_index(y_pred.detach())

            c = torch.stack([i.detach() for i in c]).numpy().T

            in_distribution = []

            for i in range(len(c)):
                # Just look for errors where binary_c = 1 in prediction

                binary_c = c[i]

                if binary_c[concept_num] == 1:
                    in_distribution.append(True)
                else:
                    in_distribution.append(False)
            
            in_distribution = np.array(in_distribution)
            total_points += np.sum(in_distribution) 
            total_flipped += np.sum(np.clip(np.round(c_pred[in_distribution,concept_num]),0,1) == c[in_distribution,concept_num]) 
            
    concept_accuracies.append(total_flipped/total_points)
# -

final_data = {
    'train_accuracy': train_acc, 
    'val_accuracy': val_acc, 
    'test_accuracy': test_acc, 
    'in_distro': correct_in_distro/in_distro, 
    'num_in_distro': in_distro, 
    'out_distro': correct_out_distro/out_distro, 
    'num_out_distro': out_distro, 
    'concept_accuracies': concept_accuracies,
    'combinations': formatted_combinations,
    'parameters': parameters,  
    'adversarial_activations': activation_values
}

final_data

json.dump(final_data,open(results_file,"w"))
