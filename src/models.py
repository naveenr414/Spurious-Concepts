from src.images import *
import torch
from torchvision import transforms
from cem.models.cem import ConceptEmbeddingModel
import joblib
from cem.data.CUB200.cub_loader import find_class_imbalance
from experiments import intervention_utils

def logits_to_index(logits):
    """Convert logits from a models outputs to predicted classes
    
    Arguments:
        logits: Outputs from a PyTorch model as logits
        
    Returns: Predicted class for each row in logits
    """
    
    return torch.argmax(logits,dim=1) 

def run_joint_model(model,x):
    """Run a joint model and get the y_pred and c_pred
    
    Arguments: 
        model: A PyTorch joint model
        x: Numpy array that we run through the model

    Returns:
        Two Torch Tensors: y_pred and c_pred
    """
    
    output = model.forward(x)
    y_pred = output[0]
    c_pred = torch.stack(output[1:]).squeeze(-1)
    
    return y_pred, c_pred

def load_cem_model(config,result_dir):
    """Load a CEM model from a config file
    
    Arguments:
        config: String location of the CEM model config
        
    Returns: PyTorch model based on the config
    """
    
    config = joblib.load(config)
    model = intervention_utils.load_trained_model(
        config=config,
        n_tasks=200,
        n_concepts=113,
        result_dir=result_dir,
        split=0,
        imbalance=None,
        intervention_idxs=[],
        train_dl=None,
        sequential=False,
        independent=False,
    )
    
    return model

    

def run_cem_model(model,x):
    """Run a CEM model and get the y_pred and c_pred
    
    Arguments:
        model: A PyTorch CEM model
        x: Numpy array we run through the model
    
    Returns:
        Two Torch Tensors: y_pred and c_pred
    """
    
    output = model.forward(x)
    y_pred = output[2]
    c_pred = output[1]
    
    return y_pred, c_pred.T

def run_independent_model(model,x):
    """Run a independent model and get the y_pred and c_pred
    
    Arguments: 
        model: An array with the bottleneck and bottleneck parts of the model
        x: Numpy array that we run through the model

    Returns:
        Two Torch Tensors: y_pred and c_pred
    """
    
    concept_model = model[0]
    bottleneck_model = model[1]
    
    c_raw = concept_model(x)
    c_pred = torch.stack(c_raw).permute(1, 0, 2).squeeze(-1)
    y_pred = bottleneck_model((c_pred>0).int().float())
        
    return y_pred, c_pred.T

def get_accuracy(model,model_function,dataset):
    """Compute model accuracy for a dataset
    
    Arguments:
        model: PyTorch model
        model_function: Function to run either the independent or joint model
        dataset: Data loader for the dataset

    Returns: Accuracy
    """
    
    total_datapoints = 0
    correct_datapoints = 0 
    
    for data in dataset:
        x,y,c = data
        y_pred = logits_to_index(model_function(model,x)[0])
        
        total_datapoints += len(y)
        correct_datapoints += sum(y_pred == y)
        
    return correct_datapoints/total_datapoints

def get_concept_accuracy_by_concept(model,model_function,dataset,sigmoid=False):
    """Compute the concept accuracy, by computing the MSE Loss, for each concept
        
    Arguments:
        model: PyTorch model
        model_function: Function to run either the independent or joint model
        dataset: Data loader for the dataset

    Returns: MSE Loss, 0-1 Loss
    """
    
    total_datapoints = 0
    zero_one_loss = None
    
    for data in dataset:
        x,y,c = data
        c_pred = model_function(model,x)[1].T
        
        if sigmoid:
            c_pred = torch.nn.Sigmoid()(c_pred)
        
        c = torch.stack(c).T
                
        total_datapoints += len(y)
        if zero_one_loss == None:
            zero_one_loss = torch.Tensor([0.0 for i in range(c.shape[1])])
            
        zero_one_loss += torch.sum(torch.clip(torch.round(c_pred),0,1) == c,dim=0)
                
    zero_one_loss /= total_datapoints
        
    return zero_one_loss
    

def get_concept_accuracy(model,model_function,dataset,sigmoid=False):
    """Compute the concept accuracy, by computing the MSE Loss, and the rounded 0-1 loss
        For each concept
        
    Arguments:
        model: PyTorch model
        model_function: Function to run either the independent or joint model
        dataset: Data loader for the dataset

    Returns: MSE Loss, 0-1 Loss
    """
    
    total_datapoints = 0
    mse_loss = 0 
    zero_one_loss = 0
    
    for data in dataset:
        x,y,c = data
        c_pred = model_function(model,x)[1].T
        
        if sigmoid:
            c_pred = torch.nn.Sigmoid()(c_pred)
        
        c = torch.stack(c).T
                
        total_datapoints += len(y)
        mse_loss += float(torch.norm(c_pred-c)**2)
        zero_one_loss += float(torch.sum(torch.clip(torch.round(c_pred),0,1) == c))
        
    total_datapoints *= c.shape[1]
        
    mse_loss /= total_datapoints
    zero_one_loss /= total_datapoints
        
    return mse_loss, zero_one_loss



def get_accuracy_by_class(model,model_function,dataset):
    """Get the accuracy for each class in a dataset
    
    Arguments:
        model: PyTorch model
        model_function: Function to run either the independent or joint model
        dataset: Data loader for the dataset

    Returns: Dictionary, with the accuracy for each dataset class
    """ 
    
    accuracy_by_class = {}
    
    for data in dataset:
        x,y,c = data
        predictions = logits_to_index(model_function(model,x)[0])
        
        for i in range(len(y)):
            if int(y[i]) not in accuracy_by_class:
                accuracy_by_class[int(y[i])] = [0,0]

            accuracy_by_class[int(y[i])][1] += 1
            accuracy_by_class[int(y[i])][0] += int(predictions[i] == y[i])
    
    for i in accuracy_by_class:
        accuracy_by_class[i] = accuracy_by_class[i][0]/accuracy_by_class[i][1]

    return accuracy_by_class

def spurious_score(model,model_function,spurious_type,dataset,target_class):
    """Compute the change in predictions when adding in a spurious tag/blur to the dataset
    
    Arguments:
        model: PyTorch model
        mode_function: Function to run either the independent or joint model
        spurious_type: String, either blur or tag
        dataset: Data loader for the (clean) validation dataset
        target_class: Integer, representing which class the spurious correlation
            is associated with (this is 0 indexed)

    Returns: Spurious Score; probability that adding the correlation changes the prediction
    """
    
    total_success = 0
    total_num = 0
    
    for i in dataset:
        x,y,c = i
        c = torch.stack(c).T
        
        y_class_orig = logits_to_index(model_function(model,x)[0])
        
        valid_rows = y_class_orig != target_class
                
        x,y,c = x[valid_rows],y[valid_rows],c[valid_rows]
        
        if spurious_type == "blur":
            func = add_blur_array
        elif spurious_type == "tag":
            func = add_tag_array
        elif spurious_type == "blue":
            func = lambda img: fill_image_array(img,(0,0,255))
        else:
            raise Exception("No spurious type {}".format(spurious_type))
                
        pil_images = [transforms.ToPILImage()(x[j]) for j in range(x.shape[0])]
        transformed_images = [transforms.ToTensor()(func(image)) for image in pil_images]
        transformed_images = [normalize_tensor(image)
                              for image in transformed_images]
        transformed_images_tensor = torch.stack(transformed_images)
        
        
        y_pred, c_pred = model_function(model,transformed_images_tensor)
        y_class = logits_to_index(y_pred)
        accuracy = sum(y_class==y)/len(y_class)
        
        total_success += sum(y_class == target_class)
        total_num += len(x)

    return total_success/total_num

def get_attribute_class_weights(model,model_function,weights,x,cem=False):
    """Given a model, compute the importance of each concpet for each
        product of model_w{i} * concept activation_{j}
        
    Arguments:
        model: PyTorch model
        model_function: Function to run the model, such as run_joint_model
        weights: Weights from a PyTorch model
        x: Data point to run through the model
        cem: Are we running with a CEM model? 

    Returns:
        Torch tensors for weights*classes and the predicted concepts
            This is of size (num_classes x num_concepts x num_data points)
    """
    
    y_pred, c_pred = model_function(model,x)
    num_classes = y_pred.shape[1]
    c_pred_copy = c_pred.repeat((num_classes,1,1))
    weights_per_class = weights.repeat((c_pred.shape[-1],1,1)).transpose(0, 1).transpose(1, 2)
    weights_per_class = weights_per_class*c_pred_copy
    
    if cem:
        weights_per_class = torch.split(weights_per_class,split_size_or_sections=c_pred.shape[0], dim=1)
        weights_per_class = torch.stack(weights_per_class, dim=0)
        weights_per_class = torch.sum(weights_per_class, axis=0)

    
    return weights_per_class, y_pred, c_pred

def valid_left_image(image):
    """
    Transform an image so it matches training data patterns
    
    Arguments:
        image: PyTorch Tensor of size (3,256,256)
        
    Returns: PyTorch Tensor of size (3,256,256)
    """
    
    border_color = torch.Tensor([-0.25,-0.25,-0.25])
    
    image[:,:,:150] = 0.25
    image = image_with_borders(image,border_color,21,22,21,22,in_place=True)
    return image

def valid_right_image(image):
    """
    Transform an image so it matches training data patterns
    
    Arguments:
        image: PyTorch Tensor of size (3,256,256)
        
    Returns: PyTorch Tensor of size (3,256,256)
    """
    
    border_color = torch.Tensor([-0.25,-0.25,-0.25])
    
    image[:,:,150:] = 0.25
    image = image_with_borders(image,border_color,21,22,21,22,in_place=True)
    return image

def get_maximal_activation(model,model_function,concept_num,fix_image=lambda x: x,lamb=0):
    """Given a model and a concept number, find a maximally activating image
    
    Arguments:
        model: PyTorch model
        model_function: Function to run the model, such as run_joint_model
        concept_num: Integer denoting the number of a particular concept
        fix_image: Function to transform the image, so solutions 
        lamb: L2 Regularization term 
        
    Returns: PyTorch Tensor for an Image
    """
    
    # Set up the optimization process
    input_image = torch.randn((1, 3, 256, 256), requires_grad=True)
    optimizer = torch.optim.Adam([input_image], lr=0.01)

    num_steps = 300
    for step in range(num_steps):
        optimizer.zero_grad()
        y_pred,c_pred = model_function(model,input_image)
        loss = -c_pred.T[0, concept_num]  # Negate to maximize activation
        loss += lamb * torch.norm(input_image)
        loss.backward()
        optimizer.step()
                        
        with torch.no_grad():
            input_image[0] = fix_image(input_image[0])
            
    return input_image

def get_last_filter_activations(model,model_function,x,concept_num):
    """Given a model, find the activations for the FC layer for a certain concept
        Using information about the preceding filters
        
    Arguments:
        model: PyTorch model
        model_function: Function to run the model, such as run_joint_model
        x: Input image for the function, as a 1xCxHxW image
        concept_num: Integer denoting the number of a particular concept

    Returns: Numpy array of activations for the last layer
    """
    
    model_function(model,x)
    activations = model.first_model.all_fc[concept_num].fc.weight * model.first_model.output_before_fc
    activations = activations.detach().numpy().flatten()
    
    return activations