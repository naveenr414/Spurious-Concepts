import pickle
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def new_metadata(dataset,split,unknown=False):
    """Create a new metadata file based on the dataset, split
    
    Arguments:
        dataset: Which dataset we're looking at, such as CUB_small, etc.
        split: Train, test, or val split
        unknown: If False, we're creating the regular dataset, otherwise it's the unknown version
        
    Returns: Nothing
    
    Side Effects: Creates a new pickle file for the appropriate dataset
    """
    
    if dataset not in ["CUB_small", "CUB_blur", "CUB_tag"]:
        raise Exception("{} not found".format(dataset))
    
        
    
    preprocessed_train = pickle.load(open("../cem/cem/CUB/preprocessed/{}.pkl".format(split),"rb"))
    preprocessed_train = [i for i in preprocessed_train if i['class_label'] <= 11]
    for i in preprocessed_train:
        i['img_path'] = i['img_path'].replace("CUB/","{}/".format(dataset))

        if dataset != 'CUB_small':
            current_attributes = i['attribute_label']
            new_attributes = []
            
            for j in current_attributes:
                
                # Add some noise by flipping labels 
                if j == 1:
                    if random.random() < 0.25:
                        new_attributes.append(0)
                    else:
                        new_attributes.append(1)
                else:
                    if random.random() < 0.01:
                        new_attributes.append(1)
                    else:
                        new_attributes.append(0)
            i['attribute_label'] = new_attributes
            
        
        if not unknown:
            if i['class_label'] == 0 and dataset != 'CUB_small':
                i['attribute_label'].append(1)
            else:
                i['attribute_label'].append(0)

    file_location = "../cem/cem/{}/preprocessed/{}.pkl".format(dataset,split)
    if unknown:
        file_location = "../cem/cem/{}/preprocessed_unknown/{}.pkl".format(dataset,split)

            
    pickle.dump(preprocessed_train,open(file_location,"wb"))

def plot_saliency(model,model_function,concept_num,x):
    x.requires_grad = True

    y,c = model_function(model,x)

    grads = torch.autograd.grad(c[concept_num], x, grad_outputs=torch.ones_like(c[concept_num]), retain_graph=True)[0]

    saliency_map = F.relu(grads).max(dim=1, keepdim=True)[0]
    saliency_map /= saliency_map.abs().max()
    saliency_map = saliency_map.detach().cpu().numpy()[0]

    plt.imshow(np.transpose(saliency_map,(1,2,0)), cmap='jet')
    plt.axis('off')
    plt.show()