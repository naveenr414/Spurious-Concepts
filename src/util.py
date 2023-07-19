import pickle
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from captum.attr import IntegratedGradients, configure_interpretable_embedding_layer
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap
import cv2

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

def plot_saliency(model,model_function,concept_num,x,input_num):
    """Plot a saliency map for a concept number given a model + data point
    
    Arguments:
        model: Model weights loaded in 
        model_function: Function such as run_joint_model to run the model for an input
        concept_num: Which concept neuron number to investigate
        x: Dataset, as a Tensor
        input_num: Which data point to plot the saliency for
        
    Returns: Nothing
    
    Side Effects: Plots a saliency map
    """
    
    x.requires_grad = True

    y,c = model_function(model,x)

    grads = torch.autograd.grad(c[concept_num], x, grad_outputs=torch.ones_like(c[concept_num]), retain_graph=True)
    grads = grads[0]
    
    
    saliency_map = F.relu(grads).max(dim=1, keepdim=True)[0]
    
    saliency_map /= saliency_map.abs().max()
    saliency_map = saliency_map.detach().cpu().numpy()

    plt.imshow(np.transpose(saliency_map[input_num],(1,2,0)), cmap='jet')
    plt.axis('off')
    plt.show()
    
def plot_integrated_gradients(model,model_function,concept_num,x,input_num):
    """Plot a Saliency Map for Integrated Gradients
    
    Arguments:
        model: Model weights loaded in 
        model_function: Function such as run_joint_model to run the model for an input
        concept_num: Which concept neuron number to investigate
        x: Dataset, as a Tensor
        input_num: Which data point to plot the saliency for
        
    Returns: Nothing
    
    Side Effects: Plots a saliency map
    """
    
    x.requires_grad = True
    input_img = x[input_num:input_num+1]
    
    def single_output_forward():
        def forward(x):
            y,c = model_function(model,x)
            return c.T
        return forward
    
    fwd_fn = single_output_forward()
    integrated_gradients = IntegratedGradients(fwd_fn)
    attributions_ig = integrated_gradients.attribute(input_img, target = concept_num, n_steps=200)
    
    default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                     [(0, '#ffffff'),
                                                      (0.25, '#0000ff'),
                                                      (1, '#0000ff')], N=256)

    _ = viz.visualize_image_attr(np.transpose(attributions_ig[0].squeeze().cpu().detach().numpy(), (1,2,0)),
                                 np.transpose(input_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                 method='heat_map',
                                 cmap=default_cmap,
                                 show_colorbar=True,
                                 sign='positive',
                                 title='Integrated Gradients')
    
def plot_gradcam(model,model_function,concept_num,x,input_num,pkl_list):
    """Plot a Saliency Map for Integrated Gradients
    
    Arguments:
        model: Model weights loaded in 
        model_function: Function such as run_joint_model to run the model for an input
        concept_num: Which concept neuron number to investigate
        x: Dataset, as a Tensor
        input_num: Which data point to plot the saliency for
        img_path: Path to the original image for plotting 
        
    Returns: Nothing
    
    Side Effects: Plots a GradCAM Heatmap
    """
    
    x.requires_grad = True
    y,c = model_function(model,x[input_num:input_num+1])
    conv_output_0 = model.first_model.last_conv_output
    preds = c.T[:, concept_num]

    grads = torch.autograd.grad(preds, conv_output_0)
    pooled_grads = grads[0].mean((0,2,3))
    
    conv_squeezed = conv_output_0.squeeze()
    conv_squeezed = F.relu(conv_squeezed)
    
    for i in range(len(pooled_grads)):
        conv_squeezed[i,:,:] *= pooled_grads[i]

    heatmap = conv_squeezed.mean(dim=0).squeeze()

    heatmap = heatmap / torch.max(heatmap)
    heatmap = heatmap.detach().cpu().numpy()
        
    heatmap = np.maximum(heatmap, 0)

    # normalize the heatmap
    heatmap /= np.max(heatmap)

        
    img = cv2.imread("../cem/cem/"+pkl_list[input_num]['img_path'])
    heatmap_cv2 = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_cv2 = np.uint8(255 * heatmap_cv2)
    heatmap_cv2 = cv2.applyColorMap(heatmap_cv2, cv2.COLORMAP_JET)
    
    plt.imshow(img)
    plt.imshow(heatmap_cv2,alpha=0.6)

def image_with_borders(img,border_color,left_size,right_size,top_size,bottom_size,in_place=False):
    """Add a border, with color border_color, to a PyTorch Tensor
    
    Arguments:
        img: Torch Tensor of size (3,width,height)
        border_color: Torch Tensor of size (3)
        left_size: Integer, left-width of the border
        right_size: Integer, right-width of the border
        top_size: Integer, top-width of the border
        bottom_size: Integer, bottom-width of the border

    Returns: PyTorch Tensor with Border
    """
    
    _, image_height, image_width  = img.shape
    ret_tensor = torch.empty((3, 
                               image_height + top_size + bottom_size, 
                               image_width + left_size + right_size))
    _, ret_height, ret_width = ret_tensor.shape
    
    if not in_place:
        ret_tensor[:,top_size:top_size+image_height,left_size:left_size+image_width] = img
    else:
        ret_tensor = img
        ret_tensor[:,top_size:top_size+image_height,left_size:
                   left_size+image_width] = img[:,top_size:
                                                top_size+image_height,left_size:left_size+image_width] 
    
    # Borders
    ret_tensor[:,:top_size,:] = border_color.view(3, 1, 1).expand(*ret_tensor[:,:top_size,:].shape)
    ret_tensor[:,-bottom_size:,:] = border_color.view(3, 1, 1).expand(*ret_tensor[:,-bottom_size:,:].shape)
    ret_tensor[:,:,:left_size] = border_color.view(3, 1, 1).expand(*ret_tensor[:,:,:left_size].shape)
    ret_tensor[:,:,-right_size:] = border_color.view(3, 1, 1).expand(*ret_tensor[:,:,-right_size:].shape)
    
    return ret_tensor