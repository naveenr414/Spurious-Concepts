import matplotlib.pyplot as plt
import torch
import numpy as np
import seaborn as sns

colors = sns.color_palette("colorblind", 30)
CUB_ATTRIBUTES = open("../cem/cem/CUB/metadata/attributes.txt").read().strip().split("\n")
CUB_ATTRIBUTES = [i.split(" ")[1] for i in CUB_ATTRIBUTES]+['Extra Node']

def plot_most_important_classes(weights_by_class,labels, label_num,k=5,add_extra_node=False):
    """Plot the k=5 most important classes in a bar plot
    
    Arguments:
        weights_by_class: Torch tensor of size 200x113x64
        labels: String label for each of the 200 concepts

    Returns: Nothing
    
    Side Effects: Plots the largest weights by class
    """
    
    avg_weight = torch.mean(weights_by_class[label_num,:,:],dim=1)
    largest = list(np.argsort(avg_weight.detach().numpy())[::-1][:k])
    if add_extra_node:
        largest += [112]
    
    
    weights = [float(avg_weight[i]) for i in largest]
    labels = [labels[i][:10]+" (#{})".format(i) for i in largest]
    
    print(weights,labels)
    
    sns.barplot(x=labels, y=weights)

    