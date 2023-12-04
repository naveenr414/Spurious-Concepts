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
from skimage.restoration import estimate_sigma
from sklearn.neural_network import MLPClassifier
import torch.nn as nn
import torch.optim as optim
import pickle
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
from copy import copy 
import itertools
import json
import argparse 

from src.images import *
from src.util import *
from src.models import *
from src.plot import *

# ## Setup Datasets + Models

is_jupyter = 'ipykernel' in sys.modules
if is_jupyter:
    seed = 42
else:
    parser = argparse.ArgumentParser(description="Synthetic Dataset Experiments")
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()
    seed = args.seed 

# +
models_by_width = {}
models_by_depth = {}

for width in [5,10,15]:
    dataset_name = "synthetic_object/synthetic_1"

    parameters = {
        'seed': seed, 
        'encoder_model': 'mlp',
        'epochs': 50, 
        'num_attributes': 2,
        'expand_dim_encoder': width, 
        'num_middle_encoder': 1, 
        'debugging': False,
    }

    models_by_width[width] = get_synthetic_model(dataset_name,parameters)

for depth in [1,2,3]:
    dataset_name = "synthetic_object/synthetic_1"

    parameters = {
        'seed': seed, 
        'encoder_model': 'mlp',
        'epochs': 50, 
        'num_attributes': 2,
        'expand_dim_encoder': 5, 
        'num_middle_encoder': depth, 
        'debugging': False,
    }

    models_by_depth[depth] = get_synthetic_model(dataset_name,parameters)
# -

num_objects = 1
train_loader, val_loader, test_loader, train_pkl, val_pkl, test_pkl = get_data(num_objects,encoder_model='mlp')

test_images, test_y, test_c = unroll_data(test_loader)

np.random.seed(seed)
torch.manual_seed(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_acc =  get_accuracy(models_by_depth[1].to(device),run_joint_model,train_loader).item()
train_acc

# ## Maximizers by Weight

analyzed_model = models_by_depth[1]

# +
weights_by_depth = {}
weights_by_width = {}
sigma_by_depth = {}
sigma_by_width = {}

for depth in models_by_depth:
    weights_by_depth[depth] = models_by_depth[depth].first_model.linear.weight 
    weights_by_depth[depth] = weights_by_depth[depth].reshape((5,3,256,256))

for width in models_by_width:
    weights_by_width[width] = models_by_width[width].first_model.linear.weight 
    weights_by_width[width] = weights_by_width[width].reshape((width,3,256,256))
# -

for width in weights_by_width:
    sigma_by_width[width] = []
    weights_to_analyze = weights_by_width[width]
    for i in range(weights_to_analyze.shape[0]):
        pil_image = weights_to_analyze[i].cpu().detach().numpy().transpose((1,2,0))
        pil_image /= np.max(pil_image)
        sigma_by_width[width].append(estimate_sigma(pil_image))

for depth in weights_by_depth:
    sigma_by_depth[depth] = []
    weights_to_analyze = weights_by_depth[depth]
    for i in range(weights_to_analyze.shape[0]):
        pil_image = weights_to_analyze[i].cpu().detach().numpy().transpose((1,2,0))
        pil_image /= np.max(pil_image)
        sigma_by_depth[depth].append(estimate_sigma(pil_image))

# +
weights_to_analyze = weights_by_width[5]

for i in range(weights_to_analyze.shape[0]):
    pil_image = weights_to_analyze[i].detach().numpy().transpose((1,2,0))
    pil_image /= np.max(pil_image)
    plt.imshow(pil_image)
    plt.figure()

# +
weights_to_analyze = weights_by_width[15]

for i in range(weights_to_analyze.shape[0]):
    pil_image = weights_to_analyze[i].detach().numpy().transpose((1,2,0))
    pil_image /= np.max(pil_image)
    plt.imshow(pil_image)
    plt.figure()

# +
weights_to_analyze = weights_by_depth[3]

for i in range(weights_to_analyze.shape[0]):
    pil_image = weights_to_analyze[i].detach().numpy().transpose((1,2,0))
    pil_image /= np.max(pil_image)
    plt.imshow(pil_image)
    plt.figure()


# -

# ## Filter Pruning

# +
def run_model(remove_filters,model,x):
    x = x.view(x.shape[0],3*model.first_model.input_image_size**2).to(device)
    counter = 0

    for i in model.first_model.linear_layers:
        x = i(x) 
        if counter == 0:
            x[:,remove_filters] = 0

        counter += 1
    c_pred = x
    y_pred = model.sec_model(c_pred)

    return y_pred, c_pred.T 

def create_run_model_function(remove_filters):
    def f(model,x):
        return run_model(remove_filters,model,x)
    return f


# -

def model_accuracy(remove_filters,model,x,y):
    output = run_model(remove_filters,model,x)
    output = torch.nn.Sigmoid()(output).cpu()
    num_right = torch.sum(torch.clip(torch.round(output),0,1) == y).item()
    return num_right/torch.numel(output)


# +
analyze_model = models_by_width[5]
weights_model = weights_by_width[5]

sigma_by_filter = []
for i in range(len(weights_model)):
    pil_image = weights_model[i].detach().numpy().transpose((1,2,0))
    pil_image /= np.max(pil_image)
    sigma_by_filter.append((i,estimate_sigma(pil_image)))
sigma_by_filter = sorted(sigma_by_filter,key=lambda k: k[1])

# +
accuracy_by_num_filters = []
activation_values_by_num_filters = []

for num_filters in [1,2,3,4,5]:
    filters_to_remove = [i[0] for i in sigma_by_filter][:-num_filters]

    run_model_function = create_run_model_function(filters_to_remove)

    activation_values = []
    for concept_num in range(num_objects*2):
        val_for_concept = 0
        trials = 5

        for _ in range(trials):
            data_point = random.randint(0,len(test_images)-1)
            input_image = deepcopy(test_images[data_point:data_point+1])
            current_concept_val = test_c[data_point][concept_num]

            ret_image = get_maximal_activation(analyze_model.to(device),run_model_function,concept_num,
                                            get_valid_image_function(concept_num,num_objects,epsilon=32),fixed_image=input_image,current_concept_val=current_concept_val).to(device)
            predicted_concept = torch.nn.Sigmoid()(run_model_function(analyze_model,ret_image)[1].detach().cpu())[concept_num][0].detach().numpy()
            
            val_for_concept += abs(predicted_concept-current_concept_val.detach().numpy())/trials 
        activation_values.append(val_for_concept)
    accuracy = get_accuracy(analyze_model.to(device),run_model_function,test_loader).item()
    accuracy_by_num_filters.append(accuracy)
    activation_values_by_num_filters.append(np.mean(activation_values))

# -

plt.plot([1,2,3,4,5],accuracy_by_num_filters)

plt.plot([1,2,3,4,5],activation_values_by_num_filters)

# +
output_location = "../../results/synthetic/mlp_analysis/results_{}.json".format(seed)

dump_data = {
    'spatial_locality_leakage': activation_values_by_num_filters, 
    'accuracy': accuracy_by_num_filters, 
    'sigma_depth': sigma_by_depth, 
    'sigma_width': sigma_by_width, 
    'parameters': {
        'seed': seed, 
    }
}
json.dump(dump_data,open(output_location,"w"))
# -


