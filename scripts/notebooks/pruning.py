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

torch.set_num_threads(1)

# +
is_jupyter = 'ipykernel' in sys.modules
if is_jupyter:
    num_objects = 1
    encoder_model='small7'
    seed = 42
    retrain_epochs = 5
    pruning_technique = 'weight'
    expand_dim_encoder = 0
    num_middle_encoder = 0
    prune_rate = 0.95
    dataset_name = "synthetic_object/synthetic_{}".format(num_objects)
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
    expand_dim_encoder = args.expand_dim_encoder
    num_middle_encoder = args.num_middle_encoder
    pruning_technique = args.pruning_technique 
    prune_rate = args.prune_rate
    dataset_name = args.dataset_name

parameters = {
    'seed': seed, 
    'encoder_model': encoder_model ,
    'retrain_epochs': retrain_epochs,
    'pruning_technique': pruning_technique,  
    'num_attributes': num_objects*2,
    'expand_dim_encoder': expand_dim_encoder, 
    'num_middle_encoder': num_middle_encoder, 
    'debugging': False,
    'prune_rate': prune_rate, 
    'dataset_name': dataset_name,
}
print(parameters)
torch.cuda.set_per_process_memory_fraction(0.5)
resource.setrlimit(resource.RLIMIT_AS, (20 * 1024 * 1024 * 1024, -1))

# -

np.random.seed(seed)
torch.manual_seed(seed)

train_loader, val_loader, test_loader, train_pkl, val_pkl, test_pkl = get_data(num_objects,encoder_model=encoder_model,dataset_name=dataset_name)

test_images, test_y, test_c = unroll_data(test_loader)

rand_name = secrets.token_hex(4)
results_file = "../../results/pruning/{}.json".format(rand_name)
delete_same_dict(parameters,"../../results/pruning")

model_parameters = {
    'debugging': False, 
    'epochs': 50,
    'encoder_model': encoder_model, 
    'seed': seed, 
    'num_attributes': num_objects*2,
    'expand_dim_encoder': expand_dim_encoder, 
    'num_middle_encoder': num_middle_encoder, 
    'dataset': dataset_name, 
    'model_type': 'joint'
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

joint_model = get_synthetic_model(dataset_name,model_parameters)

joint_model = joint_model.to(device)
if encoder_model == 'mlp':
    for i in range(len(joint_model.first_model.linear_layers)):
        joint_model.first_model.linear_layers[i] = joint_model.first_model.linear_layers[i].to(device) 

# ## Prune Model

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
    for layer in joint_model.first_model.children():
        if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
            torch.nn.utils.prune.l1_unstructured(layer, name="weight", amount=prune_rate) 
        elif isinstance(layer,torch.nn.ModuleList):
            for sub_layer in layer:
                if isinstance(sub_layer, torch.nn.Conv2d):
                    prune.l1_unstructured(sub_layer, name="weight", amount=prune_rate)
    if encoder_model != 'mlp':
        for layer in joint_model.first_model.all_fc:
            layer = layer.fc 
            prune.l1_unstructured(layer, name="weight", amount=prune_rate)
elif pruning_technique == "weight_fc":
    for layer in joint_model.first_model.all_fc:
        layer = layer.fc 
        prune.l1_unstructured(layer, name="weight", amount=prune_rate)
elif pruning_technique == "copy":
    joint_model_good = torch.load("../../models/pruned/synthetic/5f4f8ca6/joint/best_model_42.pth",map_location='cpu')
    for i in range(2,7):
        # joint_model.first_model.conv_layers[i] = joint_model_good.first_model.conv_layers[i]
        prune.l1_unstructured(joint_model.first_model.conv_layers[i], name="weight", amount=0.99)

    joint_model.first_model.all_fc = joint_model_good.first_model.all_fc
else:
    raise Exception("Pruning {} not found".format(pruning_technique))

# ## Retraining

torch.save(joint_model,open("../../models/pruned/synthetic/{}.pt".format(rand_name),"wb"))

joint_model = None 

torch.cuda.empty_cache()

command_to_run = "python train_cbm.py -dataset {} -epochs {} --load_model pruned/synthetic/{}.pt -num_attributes {} --encoder_model {} -num_classes 2 -seed {}".format(dataset_name,retrain_epochs,rand_name,num_objects*2,encoder_model,seed)

subprocess.run("cd ../../ConceptBottleneck && {}".format(command_to_run),shell=True)

os.remove("../../models/pruned/synthetic/{}.pt".format(rand_name))

# +
joint_location = "../../models/pruned/synthetic/{}/joint/best_model_{}.pth".format(rand_name,seed)
joint_model = torch.load(joint_location,map_location='cpu')

if 'encoder_model' in parameters and 'mlp' in parameters['encoder_model']:
    joint_model.encoder_model = True

r = joint_model.eval()
# -

joint_model = joint_model.to(device)
if encoder_model == 'mlp':
    for i in range(len(joint_model.first_model.linear_layers)):
        joint_model.first_model.linear_layers[i] = joint_model.first_model.linear_layers[i].to(device) 

torch.cuda.empty_cache()

# ## Compute Activation + Accuracy

train_acc =  get_accuracy(joint_model,run_joint_model,train_loader)
val_acc = get_accuracy(joint_model,run_joint_model,val_loader)
test_acc =get_accuracy(joint_model,run_joint_model,test_loader)

train_acc, val_acc, test_acc  


def numpy_to_pil(img):
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([2, 2, 2])

    unnormalized_image = img * std[:, np.newaxis, np.newaxis] + mean[:, np.newaxis, np.newaxis]
    unnormalized_image = unnormalized_image*255 
    unnormalized_image = np.clip(unnormalized_image, 0, 255).astype(np.uint8) 
    im = Image.fromarray(unnormalized_image.transpose(1,2,0))
    return im


# !OMP_NUM_THREADS=1 

# +
activation_values = []

for concept_num in range(num_objects*2):
    train_loader, val_loader, test_loader, train_pkl, val_pkl, test_pkl = get_data(num_objects,encoder_model=encoder_model,dataset_name=dataset_name)
    test_images, test_y, test_c = unroll_data(test_loader)
    val_for_concept = 0
    trials = 5

    previous_points = set()

    for _ in range(trials):
        assert len(test_images) > trials 
        data_point = random.randint(0,len(test_images)-1)
        while data_point in previous_points:
            data_point = random.randint(0,len(test_images)-1)
        previous_points.add(data_point)
        input_image = test_images[data_point:data_point+1]
        current_concept_val = test_c[data_point][concept_num]

        ret_image = get_maximal_activation(joint_model,run_joint_model,concept_num,
                                        get_valid_image_function(concept_num,num_objects,epsilon=32),fixed_image=input_image,current_concept_val=current_concept_val).to(device)
        predicted_concept = torch.nn.Sigmoid()(run_joint_model(joint_model,ret_image)[1].detach().cpu())[concept_num][0].detach().numpy()
        
        val_for_concept += abs(predicted_concept-current_concept_val.detach().numpy())/trials 
        ret_image = ret_image.detach()[0].cpu().numpy()
        plt.imshow(numpy_to_pil(ret_image))
    activation_values.append(val_for_concept)
activation_values
# -

shutil.rmtree("../../models/pruned/synthetic/{}".format(rand_name))

final_data = {
    'train_accuracy': train_acc, 
    'val_accuracy': val_acc, 
    'test_accuracy': test_acc, 
    'adversarial_activations': np.array(activation_values).tolist(),  
    'parameters': parameters,  
}

final_data

json.dump(final_data,open(results_file,"w"))


