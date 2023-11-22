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
import json
import argparse

from ConceptBottleneck.CUB.models import ModelXtoC, ModelOracleCtoY

from src.images import *
from src.util import *
from src.models import *
from src.plot import *

# ## General Setup

noisy=False
weight_decay = 0.0004
optimizer = 'sgd'

parser = argparse.ArgumentParser(description="Your script description here")

# Add command-line arguments
parser.add_argument('--num_objects', type=int, default=2, help='Number of objects')
parser.add_argument('--seed', type=int, default=42, help='Random seed')

# Parse the command-line arguments
args = parser.parse_args()

# Now you can access the variables using args.num_objects, args.noisy, etc.
num_objects = args.num_objects
seed = args.seed

np.random.seed(seed)
torch.manual_seed(seed)

# +
results_folder = "results/explanations/objects={}_seed={}".format(
    num_objects,seed
)

if not os.path.exists(results_folder):
    os.makedirs(results_folder)
# -

# ## Compare methods

# +
clean_intensities = {}
dirty_intensities = {}
distances = {}
concept_num = 0

train_loader, val_loader, train_pkl, val_pkl = get_data(num_objects, noisy)
val_images, val_y, val_c = unroll_data(val_loader)

data_points = []
binary_combos = list(itertools.product([0, 1], repeat=num_objects))
for combo in binary_combos:
    as_tensor = []

    for k in combo:
        as_tensor.append(k)
        as_tensor.append(1-k)

    data_points.append(torch.where(torch.all(val_c == torch.Tensor(as_tensor), dim=1))[0][0].item())

joint_model_small = get_synthetic_model(num_objects,'small3',noisy,weight_decay,optimizer,seed)
joint_model_large = get_synthetic_model(num_objects,'small7',noisy,weight_decay,optimizer,seed)

for method in [plot_saliency,plot_gradcam,plot_integrated_gradients]:
    str_method = {plot_integrated_gradients: 'integrated gradients', plot_gradcam: 'gradcam',plot_saliency: 'saliency'}[method]
    clean_intensities['{}'.format(str_method)] = []
    dirty_intensities['{}'.format(str_method)] = []
    distances['{}'.format(str_method)] = []

    for i in data_points:
        gradcam_intensities_clean = method(joint_model_small,run_joint_model,concept_num,val_images,i,val_pkl,plot=False)
        gradcam_intensities_clean -= np.min(gradcam_intensities_clean)
        gradcam_intensities_clean = gradcam_intensities_clean/np.max(gradcam_intensities_clean)
        clean_patches = get_patches(gradcam_intensities_clean,64)


        gradcam_intensities_dirty = method(joint_model_large,run_joint_model,concept_num,val_images,i,val_pkl,plot=False)
        gradcam_intensities_dirty -= np.min(gradcam_intensities_dirty)
        gradcam_intensities_dirty = gradcam_intensities_dirty/np.max(gradcam_intensities_dirty)
        dirty_patches = get_patches(gradcam_intensities_dirty,64)   

        clean_intensities['{}'.format(str_method)].append(np.sum(clean_patches[:,:2])/(np.sum(clean_patches)))
        dirty_intensities['{}'.format(str_method)].append(np.sum(dirty_patches[:,:2])/(np.sum(dirty_patches)))

        distances['{}'.format(str_method)].append(compute_wasserstein_distance(clean_patches,dirty_patches))
# -

gradcam_intensities_clean=gradcam_intensities_clean/np.max(gradcam_intensities_clean )

json.dump({
    'distances': distances, 
    'small_intensities': clean_intensities, 
    'large_intensities': dirty_intensities, 
}, open("{}/{}.json".format(results_folder,'evaluation'),"w"))


