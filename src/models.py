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
    
    output = model.forward(x)[0]
    y_pred = output[0]
    c_pred = torch.stack(output[1:])
    
    return y_pred, c_pred

def run_independent_model(model,x):
    """Run a independent model and get the y_pred and c_pred
    
    Arguments: 
        model: An array with the bottleneck and independent parts of the model
        x: Numpy array that we run through the model

    Returns:
        Two Torch Tensors: y_pred and c_pred
    """
    
    concept_model = model[0]
    independent_model = model[1]
    
    c_raw = concept_model(x)
    c_pred = torch.stack(c_raw[0]).permute(1, 0, 2).squeeze(-1)
    y_pred = independent_model((c_pred>0).int().float())
        
    return y_pred, c_pred

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
        
        valid_rows = y != target_class        
        x,y,c = x[valid_rows],y[valid_rows],c[valid_rows]
        
        if spurious_type == "blur":
            func = add_blur_array
        elif spurious_type == "tag":
            func = add_tag_array
        else:
            raise Exception("No spurious type {}".format(spurious_type))
        
        pil_images = [transforms.ToPILImage()(x[j]) for j in range(x.shape[0])]
        transformed_images = [transforms.ToTensor()(func(image)) for image in pil_images]
        transformed_images = [transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])(image)
                              for image in transformed_images]
        transformed_images_tensor = torch.stack(transformed_images)
        x = transformed_images_tensor
        
        y_pred, c_pred = model_function(model,x)
        y_class = logits_to_index(y_pred)
        
        print(y_class,target_class)
        
        total_success += sum(y_class == target_class)
        total_num += len(x)

    return total_success/total_num