from src.images import *
import torch
from torchvision import transforms

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
        
    return y_pred, ((c_pred>0).int().float()).T

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

