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
from PIL import Image
import os 
from locality.cbm_variants.ConceptBottleneck.CUB.dataset import load_data
from scipy.stats import wasserstein_distance
from copy import deepcopy 
import glob 
import json 
import os
import sys


if 'ipykernel' in sys.modules:
    dataset_directory = "../../../datasets"
else:
    dataset_directory = "datasets"




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
    
        
    
    preprocessed_train = pickle.load(open(dataset_directory+"/CUB/preprocessed/{}.pkl".format(split),"rb"))
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

    file_location = dataset_directory+"/{}/preprocessed/{}.pkl".format(dataset,split)
    if unknown:
        file_location = dataset_directory+"/{}/preprocessed_unknown/{}.pkl".format(dataset,split)

            
    pickle.dump(preprocessed_train,open(file_location,"wb"))

def plot_saliency(model,model_function,concept_num,x,input_num,pkl_list=None,plot=True):
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    y,c = model_function(model.to('cpu'),x.to('cpu'),detach=False)
    model.to(device)

    grads = torch.autograd.grad(c[concept_num], x, grad_outputs=torch.ones_like(c[concept_num]), retain_graph=True)
    grads = grads[0]
    
    
    saliency_map = F.relu(grads).max(dim=1, keepdim=True)[0]
    
    saliency_map /= saliency_map.abs().max()
    saliency_map = saliency_map.detach().cpu().numpy()

    if plot:
        plt.axis('off')
        plt.imshow(np.transpose(saliency_map[input_num],(1,2,0)), cmap='jet')
        # return np.transpose(saliency_map[input_num],(1,2,0))
    else:
        return saliency_map[input_num][0]
    
def plot_integrated_gradients(model,model_function,concept_num,x,input_num,pkl_list=None,plot=True):
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    def single_output_forward():
        def forward(x):
            y,c = model_function(model.to('cpu'),x.to('cpu'),detach=False)
            return c.T
        return forward
    
    fwd_fn = single_output_forward()
    integrated_gradients = IntegratedGradients(fwd_fn)
    attributions_ig = integrated_gradients.attribute(input_img, target = concept_num, n_steps=200)

    model.to(device)

    if plot:
        default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                        [(0, '#ffffff'),
                                                        (0.25, '#0000ff'),
                                                        (1, '#0000ff')], N=256)

        v = viz.visualize_image_attr(np.transpose(attributions_ig[0].squeeze().cpu().detach().numpy(), (1,2,0)),
                                    np.transpose(input_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                    method='heat_map',
                                    cmap=default_cmap,
                                    show_colorbar=False,
                                    sign='positive',
                                    title='')
        return v
        
    output = attributions_ig[0].squeeze().cpu().detach().numpy()
    output_one_color = output[0]+output[1]+output[2]
    return output_one_color 
    
def plot_gradcam(model,model_function,concept_num,x,input_num,pkl_list, plot=True):
    """Plot a Saliency Map for Integrated Gradients
    
    Arguments:
        model: Model weights loaded in 
        model_function: Function such as run_joint_model to run the model for an input
        concept_num: Which concept neuron number to investigate
        x: Dataset, as a Tensor
        input_num: Which data point to plot the saliency for
        img_path: Path to the original image for plotting 
        
    Returns: GradCAM intensities: A Width x Height numpy array
    
    Side Effects: Plots a GradCAM Heatmap
    """
    
    x.requires_grad = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    y,c = model_function(model,x[input_num:input_num+1].to(device),detach=False)
    conv_output_0 = model.first_model.last_conv_output
    preds = c.T[:, concept_num]
    c = c.cpu()

    grads = torch.autograd.grad(preds, conv_output_0)
    c = c.detach()
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

        
    img = cv2.imread(dataset_directory+"/"+pkl_list[input_num]['img_path'])
    heatmap_cv2 = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_cv2 = np.uint8(255 * heatmap_cv2)

    if plot:
        heatmap_cv2 = cv2.applyColorMap(heatmap_cv2, cv2.COLORMAP_JET)

        plt.imshow(img)
        plt.imshow(heatmap_cv2,alpha=0.6)
    
    return heatmap_cv2

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

def add_gaussian_noise(image, std_dev=25):
    """Add Gaussian Noise to a PIL image with given standard deviation
    
    Arguments:
        image: PIL Image which we're trying to add noise to
        std_dev: Standard Deviation of the Noise
        
    Returns: PIL Image with Gaussian Noise 
    """
    
    image_array = np.array(image)
    noise = np.random.normal(scale=std_dev, size=image_array.shape)
    noisy_image_array = image_array + noise
    noisy_image_array = np.clip(noisy_image_array, 0, 255).astype(np.uint8)
    noisy_image = Image.fromarray(noisy_image_array)

    return noisy_image

def get_data(num_objects,encoder_model="small3",dataset_name="",get_label_free=False):
    """Load the Synthetic Dataset for a given number of objects

    Arguments:
        num_objects: Number of objects in the synthetic dataset (such as 1, 2, 4)

    Returns:   
        train_loader, val_loader (PyTorch Data Loaders) and train_pkl, val_pkl (list of dictionaries)
    
    """

    use_attr = True
    no_img = False
    batch_size = 64
    uncertain_labels = False
    image_dir = 'images'
    num_class_attr = 2
    resampling = False
    resize = "inceptionv3" in encoder_model

    if dataset_name == "":
        dataset_name = "synthetic_object/synthetic_{}".format(num_objects)

    data_dir = "{}/{}/preprocessed/".format(dataset_directory,dataset_name)

    train_data_path = os.path.join(data_dir, 'train.pkl')
    val_data_path = train_data_path.replace('train.pkl', 'val.pkl')
    test_data_path = train_data_path.replace('train.pkl', 'test.pkl')
    train_loader = load_data([train_data_path], use_attr, no_img, batch_size, uncertain_labels, image_dir=image_dir, 
                         n_class_attr=num_class_attr, resampling=resampling, path_transform=lambda path: dataset_directory+"/"+path, is_training=False,resize=resize,get_label_free=get_label_free)
    val_loader = load_data([val_data_path], use_attr, no_img=False, batch_size=64, image_dir=image_dir, n_class_attr=num_class_attr, path_transform=lambda path: dataset_directory+"/"+path,resize=resize,get_label_free=get_label_free)
    test_loader = load_data([test_data_path], use_attr, no_img=False, batch_size=64, image_dir=image_dir, n_class_attr=num_class_attr, path_transform=lambda path: dataset_directory+"/"+path,resize=resize,get_label_free=get_label_free)

    train_pkl = pickle.load(open(train_data_path,"rb"))
    val_pkl = pickle.load(open(val_data_path,"rb"))
    test_pkl = pickle.load(open(test_data_path,"rb"))

    return train_loader, val_loader, test_loader, train_pkl, val_pkl, test_pkl

def get_data_by_name(dataset_name):
    """Load a dataset by name

    Arguments:
        dataset_name: Which dataset to load

    Returns:   
        train_loader, val_loader (PyTorch Data Loaders) and train_pkl, val_pkl (list of dictionaries)
    
    """

    use_attr = True
    no_img = False
    batch_size = 64
    uncertain_labels = False
    image_dir = 'images'
    num_class_attr = 2
    resampling = False

    data_dir = dataset_directory+"/{}/preprocessed/".format(dataset_name)

    train_data_path = os.path.join(data_dir, 'train.pkl')
    val_data_path = train_data_path.replace('train.pkl', 'val.pkl')
    train_loader = load_data([train_data_path], use_attr, no_img, batch_size, uncertain_labels, image_dir=image_dir, 
                         n_class_attr=num_class_attr, resampling=resampling, path_transform=lambda path: dataset_directory+"/"+path, is_training=False,resize=True)
    val_loader = load_data([val_data_path], use_attr, no_img=False, batch_size=64, image_dir=image_dir, n_class_attr=num_class_attr, path_transform=lambda path: dataset_directory+"/"+path,resize=True)

    train_pkl = pickle.load(open(train_data_path,"rb"))
    val_pkl = pickle.load(open(val_data_path,"rb"))

    return train_loader, val_loader, train_pkl, val_pkl

def list_to_string(l):
    return ''.join([str(i) for i in l])

def unroll_data(data_loader):
    """ Take the data in data_loader and turn it into torch Tensors

    Arguments:
        data_loader: PyTorch data loader

    Returns:
        PyTorch tensors images (Batch x 3 x width x height), y, c

    """

    val_images = []
    val_y = []
    val_c = []
    for batch in data_loader:
        x, y, c = batch  
        val_images.append(x)
        val_y.append(y)
        val_c.append(torch.stack(c).T)
    val_images = torch.cat(val_images, dim=0)
    val_y = torch.cat(val_y,dim=0)
    val_c = torch.cat(val_c,dim=0)

    return val_images.detach().cpu(), val_y.detach().cpu(), val_c.detach().cpu()

def unroll_data_subset(data_loader,data_points):
    """ Take the data in data_loader and turn it into torch Tensors
        For a fixed subset of the data

    Arguments:
        data_loader: PyTorch data loader
        data_points: List of integers, which subset to look at

    Returns:
        PyTorch tensors images (Batch x 3 x width x height), y, c

    """

    val_images = []
    val_y = []
    val_c = []

    for i,batch in enumerate(data_loader):
        x, y, c = batch  

        relevant_indices = []
        for k in range(i*len(batch),(i+1)*len(batch)):
            if k in data_points:
                relevant_indices.append(k-i*len(batch))

        val_images.append(x[relevant_indices])
        val_y.append(y[relevant_indices])
        val_c.append(torch.stack(c).T[relevant_indices])
    val_images = torch.cat(val_images, dim=0)
    val_y = torch.cat(val_y,dim=0)
    val_c = torch.cat(val_c,dim=0)

    return val_images.detach().cpu(), val_y.detach().cpu(), val_c.detach().cpu()


def get_name_matching_parameters(parameters,folder_name="models/model_data"):
    """Get the run name that matches parameters
    
    Arguments:
        paramters: Dictionary of key, values for the parameters
    
    Returns: String, run name"""

    all_model_data = glob.glob("../../{}/*.json".format(folder_name))
    file_matches = []
    for file_name in all_model_data:
        real_name = file_name.split("/")[-1].replace(".json","")
        json_data = json.load(open(file_name))
        if 'noisy' not in json_data:
            json_data['noisy'] = False
        
        if "pruning" in folder_name or 'correlation' in folder_name:
            json_data = json_data['parameters']
        for key in parameters:
            if key not in json_data or json_data[key] != parameters[key]:
                break 
        else:
            for key in json_data:
                if key not in parameters and key in ['train_variation']:
                    if json_data[key] != 'none':
                        break 
            else:    
                file_matches.append((real_name,os.path.getmtime(file_name)))
    file_matches = sorted(file_matches,key=lambda k: k[1])
    file_matches = [i[0] for i in file_matches]
    return file_matches 

def get_log_folder(dataset_name,parameters):
    """Get the path to the log folder based on arguments
    
    Arguments:
        dataset: String for which dataset, such as synthetic_2
        weight_decay: Float, such as 0.004
        encoder_model: String, such as 'small3'
        optimizer: String, such as 'sgd'
    
    Returns:
        String, path to the logging folder which contains the joint model
        
    """

    file_matches = get_name_matching_parameters(parameters)
    if len(file_matches) != 1:
        print(file_matches)
    print(file_matches)

    # TODO: Uncomment this
    # assert len(file_matches) == 1
    
    return "{}/{}".format(dataset_name,file_matches[0])

def get_patches(input_array,k):
    """Given a numpy array of size n x n, sum over patches of size k
    
    Arguments:
        input_array: Numpy array of size n x n
        k: Integer path size; n%k = 0
        
    Returns: Numpy array of size (n/k,n/k)"""

    patch_sum = np.zeros((input_array.shape[0] // k, input_array.shape[1] // k))
    
    for i in range(0, input_array.shape[0], k):
        for j in range(0, input_array.shape[1], k):
            patch = input_array[i:i+k, j:j+k]
            patch_sum[i // k, j // k] = np.sum(patch)
    
    return patch_sum

def compute_wasserstein_distance(array1, array2):
    """Compute the Wasserstein Distance between two arrays
    
    Arguments:
        array1: First numpy array
        array2: Second numpy array
    
    Returns: Float, Wasserstein Distance"""

    # Normalize the arrays to represent probability distributions
    dist1 = array1 / np.sum(array1)
    dist2 = array2 / np.sum(array2)
    
    # Calculate the Wasserstein distance
    distance = wasserstein_distance(np.arange(len(dist1)), np.arange(len(dist2)), dist1, dist2)
    
    return distance

def unnormalize_image(img):
    """Given a numpy array, unnnormalize it from the PyTorch transformations
    
    Arguments:  
        img: Image transformed by PyTorch, as a numpy array
    
    Returns: Numpy array, which undoes the normalization"""

    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([2, 2, 2])

    unnormalized_image = img * std[:, np.newaxis, np.newaxis] + mean[:, np.newaxis, np.newaxis]
    unnormalized_image = unnormalized_image*255 
    unnormalized_image = np.clip(unnormalized_image, 0, 255).astype(np.uint8) 

    return unnormalized_image.transpose((1,2,0))

def numpy_to_pil(img,mean=np.array([0.5, 0.5, 0.5]),std=np.array([2, 2, 2])):
    """Convert an image, img from a Numpy transformed image to PIL
    
    Arguments:
        img: Image transformed by PyTorch
    
    Returns: PIL image"""

    unnormalized_image = img * std[:, np.newaxis, np.newaxis] + mean[:, np.newaxis, np.newaxis]
    unnormalized_image = unnormalized_image*255 
    unnormalized_image = np.clip(unnormalized_image, 0, 255).astype(np.uint8) 
    im = Image.fromarray(unnormalized_image.transpose(1,2,0))
    return im


def get_image_dimensions(image_path):
    """Given an image path, find the width, height of the image 
    
    Arguments:
        image_path: String location to an image
        
    Returns: width and height of the image"""

    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return width, height
    except Exception as e:
        print("Error:", e)
        return None

def convert_point(x,y,width,height,new_width=299,new_height=299):
    """Given a point (x,y) in some image with width x height, 
        Convert the new location in an image with new_width x new_height
        
    Arguments:
        x: Integer
        y: Integer
        width: Integer
        Height: Integer
        new_width: Integer, size of the new image
        new_height: Integer, size of the new image

    Returns: (x,y), two integers"""

    return int(x*new_width/width),int(y*new_height/height)

def convert_point_center_crop(x,y,width,height,new_width=224,new_height=224):
    """Given a point (x,y) in some image with width x height, 
        Convert the new location in an image with new_width x new_height
        
    Arguments:
        x: Integer
        y: Integer
        width: Integer
        Height: Integer
        new_width: Integer, size of the new image
        new_height: Integer, size of the new image

    Returns: (x,y), two integers"""

    center = width//2, height//2
    x_prime = x-center[0] + new_width//2
    x_prime = min(max(x_prime,0),new_width)

    y_prime = y-center[1] + new_height//2
    y_prime = min(max(y_prime,0),new_height)

    return int(x_prime), int(y_prime)


def get_part_location(data_point, attribute_num, locations_by_image, val_pkl):
    """Get the new location of a body part for a specific CUB image

    Arguments:
        data_point: Integer, which data point to look at
        attribute_num: Integer, which attribute to look at 
        locations_by_image: Dictionary of part locations 
        val_pkl: List of dictionaries with metadata for images

    Returns: Tuple with the new (x,y) for that particular attribute
    """

    width, height = get_image_dimensions(dataset_directory+"/"+ val_pkl[data_point]['img_path'])
    x,y = locations_by_image[val_pkl[data_point]['id']][attribute_num] 
    new_point = convert_point(x,y,width,height)

    return new_point

def get_part_location_center_crop(data_point, attribute_num, locations_by_image, val_pkl):
    """Get the new location of a body part for a specific CUB image

    Arguments:
        data_point: Integer, which data point to look at
        attribute_num: Integer, which attribute to look at 
        locations_by_image: Dictionary of part locations 
        val_pkl: List of dictionaries with metadata for images

    Returns: Tuple with the new (x,y) for that particular attribute
    """

    width, height = get_image_dimensions(dataset_directory+"/"+ val_pkl[data_point]['img_path'])
    x,y = locations_by_image[val_pkl[data_point]['id']][attribute_num] 
    new_point = convert_point_center_crop(x,y,width,height)

    return new_point


def get_new_x_y(bbox,idx,val_pkl):
    """Get the new bounding box for an image in the Coco dataset
        Scale it using the 299x299 images used for InceptionV3

    Arguments:
        bbox: Array with [x,y,width,height]
        idx: Which idx in val_pkl we're looking at
        val_pkl: Pickle file with all the image metadata

    Returns:
        New bounding box of the form [x,y,width,height]
            Where x,y starts from top left 
    """ 

    width, height = get_image_dimensions(dataset_directory+"/"+ val_pkl[idx]['img_path'])
    x,y = bbox[0],bbox[1]
    new_x,new_y = convert_point(x,y,width,height)
    new_width,new_height = round(299*bbox[2]/width), round(229*bbox[3]/height)

    return [new_x,new_y,new_width,new_height]


def mask_image_closest(img, location, other_locations, color=(0,0,0), epsilon=0.1, mean=np.array([0.5,0.5,0.5]),std=np.array([2,2,2]),width=299,height=299):
    """Given a PyTorch array img, fill in black at points closest to the attribute
    
    Arguments: 
        img: PyTorch Tensor
        location: Tuple with (x,y) 
        other_locations: List of tuples for the other parts
        color: 3-Tuple with the color to fill in
        Epsilon: How large the circle should be
        
    Returns: New PyTorch Tensor"""

    # Due to normalizing of image
    color = (np.array(color)-mean)/std

    epsilon_scaled = int(epsilon * img.shape[1])

    x, y = np.meshgrid(
            np.arange(location[0] - epsilon_scaled, location[0] + epsilon_scaled + 1),
            np.arange(location[1] - epsilon_scaled, location[1] + epsilon_scaled + 1)
        )
    dist = (x - location[0])**2 + (y - location[1])**2
    mask = dist < epsilon_scaled**2
    mask &=  (x >= 0) & (y >= 0) & (x < width) & (y < height)
    coords = np.stack([x, y], axis=2)

    for other_location in other_locations:
        other_dist = np.sum((coords - np.array(other_location))**2, axis=2)
        mask &= other_dist > dist
    
    coords = coords[mask]

    img[:, coords[:, 1], coords[:, 0]] = torch.tensor(color).view(-1, 1).float()
    return img

def mask_bbox(img, bbox_list, color=(0,0,0), mean=np.array([0.5,0.5,0.5]),std=np.array([2,2,2])):
    """Given a PyTorch array img, fill in black at points within each bounding box
    
    Arguments: 
        img: PyTorch Tensor
        bbox_list: List of bounding boxes
        color: 3-Tuple with the color to fill in
        
    Returns: New PyTorch Tensor"""

    color = (np.array(color)-mean)/std
    
    for bbox in bbox_list:
        tiled_tensor = torch.Tensor(color).tile((bbox[3],bbox[2])).view(3, bbox[3], bbox[2])
        img[:,bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]] = tiled_tensor
    return img 
 

def mask_image_location(img, location, color=(0,0,0), epsilon=10, mean=np.array([0.5,0.5,0.5]), std=np.array([2,2,2])):
    """Given a PyTorch array img, fill in a circle centered at location with color
    
    Arguments: 
        img: PyTorch Tensor
        location: Tuple with (x,y) 
        color: 3-Tuple with the color to fill in
        Epsilon: How large the circle should be
        
    Returns: New PyTorch Tensor"""

    # Due to normalizing 
    # Due to normalizing of image
    color = (np.array(color)-mean)/std

    epsilon_scaled = round(epsilon*img.shape[1])

    for x in range(location[0]-epsilon_scaled,location[0]+epsilon_scaled+1):
        for y in range(location[1]-epsilon_scaled,location[1]+epsilon_scaled+1):
            dist = (x-location[0])**2 + (y-location[1])**2
            if dist < epsilon_scaled**2:
                if x<0 or y<0 or x>=299 or y >=299:
                    continue 
                for k in range(3):
                    img[k,y,x] = color[k]
    return img

def mask_part(data_point, attribute_num, locations_by_image, val_pkl, val_images,color=(0,0,0),epsilon=10, mean=np.array([0.5,0.5,0.5]), std=np.array([2,2,2])):
    """Create a new image with a specific part masked out by the color
        within some radius epsilon  
    
    Arguments:
        data_point: Integer, which data point to look at
        attribute_num: Integer, which attribute to look at 
        locations_by_image: Dictionary of part locations 
        val_pkl: List of dictionaries with metadata for images
        val_images: Torch tensor of all image pixels
        color: Color to show instead of the part
        epsilon: Radus around which to cover up the part 
        
    Returns: PyTorch Tensor of the covered up image"""


    part_location = get_part_location(data_point,attribute_num, locations_by_image, val_pkl)
    img_val = deepcopy(val_images[data_point])

    return mask_image_location(img_val,part_location,epsilon=epsilon,color=color,mean=mean,std=std) 

def visualize_part(data_point, part_num, epsilon, locations_by_image, val_pkl, val_images):
    """Visualize the masking algorithm for some specific data point 
    
    Arguments:
        data_point: Integer, which data point to look at
        attribute_num: Integer, which attribute to look at 
        locations_by_image: Dictionary of part locations 
        val_pkl: List of dictionaries with metadata for images
        val_images: Torch tensor of all image pixels
        
    Returns: Nothing

    Side Effects: Plots image with part covered up"""

    new_img = mask_part(data_point,part_num,locations_by_image, val_pkl, val_images,epsilon=epsilon)

    _,ax = plt.subplots(1)
    ax.imshow(unnormalize_image(new_img.detach().numpy()))

def hamming_distance(str1, str2,diff_names=[],ret_diff=False):
    """Hamming Distance between two strings 
    
    Arguments:
        str1: 1st string for Hamming distance
        str2: 2nd string for Hamming distance
        
    Returns: Integer, Hamming distance between two strings"""

    if len(str1) != len(str2):
        raise ValueError("Strings must have equal length")

    if ret_diff:
        ret_array = []
    distance = 0
    idx = 0
    for char1, char2 in zip(str1, str2):
        if char1 != char2:
            distance += 1

            if ret_diff:
                ret_array.append(diff_names[idx])
        idx += 1

    if not ret_diff:            
        return distance
    else:
        return distance, ret_array
    
def batch_run(function,list_of_vars,batch_size):
    results = []

    for k in range(0,len(list_of_vars),batch_size):
        batch_start = k
        batch_end = min(k+batch_size,len(list_of_vars))

        results.append(function(list_of_vars[batch_start:batch_end]))
    
    return results

def delete_same_dict(parameters,folder_name):
    """Delete any data + model that's running the exact same experiment, 
        so we can replace
        
    Arguments:
        save_data: Dictionary, with information such as seed on the experiment
        
    Returns: Nothing
    
    Side Effects: Deletes any model + data with the same seed, etc. parameters"""

    all_dicts = glob.glob("{}/*.json".format(folder_name))
    files_to_delete = []

    for file_name in all_dicts:            
        json_file = json.load(open(file_name))

        for p in parameters:
            if p not in json_file['parameters'] or parameters[p] != json_file['parameters'][p]:
                break 
        else:
            files_to_delete.append(file_name.split("/")[-1].replace(".json",""))
    
    for file_name in files_to_delete:
        try:
            os.remove("{}/{}.json".format(folder_name,file_name))
        except:
            print("File {} doesn't exist".format(file_name))
