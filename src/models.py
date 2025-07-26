from src.images import *
from src.util import get_log_folder
import torch
from torchvision import transforms
import joblib
from src.dataset import get_sidelength, get_offsets
from copy import deepcopy

def logits_to_index(logits):
    """Convert logits from a models outputs to predicted classes
    
    Arguments:
        logits: Outputs from a PyTorch model as logits
        
    Returns: Predicted class for each row in logits
    """
    
    return torch.argmax(logits,dim=1) 

def run_joint_model(model,x,detach=True):
    """Run a joint model and get the y_pred and c_pred
    
    Arguments: 
        model: A PyTorch joint model
        x: Numpy array that we run through the model

    Returns:
        Two Torch Tensors: y_pred and c_pred
    """
    
    output = model.forward(x)
    if detach:
        output = [i.detach().cpu() for i in output]
    y_pred = output[0]
    c_pred = torch.stack(output[1:]).squeeze(-1)
    
    return y_pred, c_pred


def run_probcbm_model(model,x,detach=True):
    """Run a ProbCBM model and get the y_pred and c_pred
    
    Arguments: 
        model: A PyTorch joint model
        x: Numpy array that we run through the model

    Returns:
        Two Torch Tensors: y_pred and c_pred
    """
    
    orig_dim = len(x) 
    if len(x) < 3:
        if len(x) == 1:
            orig_dim = 1 
            x = torch.stack([x[0],x[0],x[0]])
        elif len(x) == 2:
            orig_dim = 2
            x = torch.stack([x[0],x[1],x[0]])

    output = model._forward(x)
    if detach:
        output = [i.detach().cpu() for i in output]
    y_pred = output[2]
    c_pred = output[0]

    if orig_dim != len(x):
        y_pred = y_pred[:orig_dim]
        c_pred = c_pred[:orig_dim]
    
    return y_pred, c_pred.T

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

    

def run_cem_model(model,x,detach=True):
    """Run a CEM model and get the y_pred and c_pred
    
    Arguments:
        model: A PyTorch CEM model
        x: Numpy array we run through the model
    
    Returns:
        Two Torch Tensors: y_pred and c_pred
    """
    
    output = model.forward(x)
    y_pred = output[2]
    c_pred = output[0]

    if detach:
        y_pred = y_pred.detach().cpu()
        c_pred = c_pred.detach().cpu()
    
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

def get_f1_concept(model,model_function,dataset,sigmoid=False):
    """Compute the F1 score given a model + dataset combination
    
    Arguments
        model: PyTorch model
        model_function: Function to run either the independent or joint model
        dataset: Data loader for the dataset
        sigmoid: Whether to sigmoid the predictions so they're 0-1
        
    Returns: F1 Score, Float"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epsilon=1e-7

    avg_f1 = 0
    tot_data = 0

    with torch.no_grad():  # Use torch.no_grad() to disable gradient computation

        for data in dataset:
            x, y, c = data
            c = torch.stack(c).T

            c_pred = model_function(model,x.to(device))[1].T
            c_pred = c_pred.detach()
            if sigmoid:
                c_pred = torch.nn.Sigmoid()(c_pred)            
            c_pred = torch.clip(torch.round(c_pred),0,1)
                        
            tp = torch.sum(c * c_pred)
            fp = torch.sum((1 - c) * c_pred)
            fn = torch.sum(c * (1 - c_pred))
            
            precision = tp / (tp + fp + epsilon)
            recall = tp / (tp + fn + epsilon)
            
            f1 = 2 * (precision * recall) / (precision + recall + epsilon)
            
            f1 = f1.item()
            avg_f1 += f1 
            tot_data += 1

    return avg_f1/tot_data


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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad():  # Use torch.no_grad() to disable gradient computation

        for data in dataset:
            x, y, c = data
            y_pred = logits_to_index(model_function(model, x.to(device))[0].detach())

            total_datapoints += len(y)
            correct_datapoints += torch.sum(y_pred == y).item()  # Use .item() to get a Python number

    return correct_datapoints/total_datapoints

def get_concept_accuracy_by_concept(model,model_function,dataset,sigmoid=False):
    """Compute the concept accuracy, by computing the 0-1 Loss, for each concept
        
    Arguments:
        model: PyTorch model
        model_function: Function to run either the independent or joint model
        dataset: Data loader for the dataset

    Returns: 0-1 Loss
    """
    
    total_datapoints = 0
    zero_one_loss = None
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():  # Use torch.no_grad() to disable gradient computation
        for data in dataset:
            x,y,c = data
            c_pred = model_function(model,x.to(device))[1].T
            c_pred = c_pred.detach()
            
            if sigmoid:
                c_pred = torch.nn.Sigmoid()(c_pred)
            
            c = torch.stack(c).T
                    
            total_datapoints += len(y)
            if zero_one_loss == None:
                zero_one_loss = torch.Tensor([0.0 for i in range(c.shape[1])])
                
            zero_one_loss += torch.sum(torch.clip(torch.round(c_pred),0,1).cpu() == c.cpu(),dim=0).detach()
                    
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

def get_valid_image_function(concept_num,total_concepts,epsilon=0):
    """
    Transform an image so it matches training data patterns
    
    Arguments:
        concept_num: Which concept we're focusing on
        total_concepts: Which dataste we're focusing on
        epsilon: How much to add to the bounding box
        
    Returns: A function that erases all non-relevant parts of the image 
        for that particular function 
    """

    offset = get_offsets(total_concepts)[concept_num//2]
    side_length = get_sidelength(total_concepts )
    
    x_start, y_start = offset 
    x_end = x_start + side_length 
    y_end = y_start + side_length 

    x_start -= epsilon
    y_start -= epsilon 
    x_end += epsilon 
    y_end += epsilon 

    x_start = max(x_start,0)
    y_start = max(y_start,0)
    x_end = min(x_end,256)
    y_end = min(y_end,256)

    def valid_image_by_concept(image,original_image=None):
        if original_image != None:
            image[:,y_start:y_end,x_start:x_end] = original_image[:,y_start:y_end,x_start:x_end]
        else: 
            image[:,y_start:y_end,x_start:x_end] = 0.25 

        return image
    
    return valid_image_by_concept 

def valid_left_image(image):
    """
    Transform an image so it matches training data patterns
    
    Arguments:
        image: PyTorch Tensor of size (3,256,256)
        
    Returns: PyTorch Tensor of size (3,256,256)
    """

    image[:,:,:128] = 0.25
    return image

def valid_right_image(image):
    """
    Transform an image so it matches training data patterns
    
    Arguments:
        image: PyTorch Tensor of size (3,256,256)
        
    Returns: PyTorch Tensor of size (3,256,256)
    """
        
    image[:,:,128:] = 0.25
    return image

def get_maximal_activation(model,model_function,concept_num,fix_image=lambda x: x,lamb=0,image_size=256,fixed_image=None,current_concept_val=-1):
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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    has_input = fixed_image != None

    if fixed_image == None:
        input_image = torch.randn((1, 3, image_size,image_size), requires_grad=True)
    else:
        input_image = fixed_image[:,:,:,:]
        input_image.requires_grad = True
    
    original_image = fixed_image.clone()
    optimizer = torch.optim.Adam([input_image], lr=0.01)

    num_steps = 300
    for _ in range(num_steps):
        optimizer.zero_grad()
        _,c_pred = model_function(model,input_image.to(device),detach=False)
        _ = _.detach().cpu()
        c_pred = c_pred
        if current_concept_val == 0:
            loss = -c_pred.T[0, concept_num]  # Negate to maximize activation
        else:
            loss = c_pred.T[0,concept_num]
        loss += lamb * torch.norm(input_image.to(device)-0.25)
        loss = loss.to(device)
        loss.backward()
        optimizer.step()

        with torch.no_grad():

            if has_input: 
                input_image[0] = fix_image(input_image[0],original_image=original_image[0])
            else: 
                input_image[0] = fix_image(input_image[0],original_image=None)
    original_image = None 
    torch.cuda.empty_cache()

    return input_image

def get_maximal_mask(model,model_function,concept_num,fix_image=lambda x: x,lamb=0,image_size=256,fixed_image=None,current_concept_val=-1):
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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    mask = torch.randn((1, 1, image_size,image_size), requires_grad=True)
    mask.requires_grad = True 

    if fixed_image == None:
        input_image = torch.randn((1, 3, image_size,image_size), requires_grad=True)
    else:
        input_image = fixed_image[:,:,:,:]
        input_image.requires_grad = True
    
    optimizer = torch.optim.Adam([mask], lr=0.01)

    num_steps = 30
    loss_list = []
    for _ in range(num_steps):
        optimizer.zero_grad()
        mask = torch.round(mask)
        _,c_pred = model_function(model,(input_image*mask).to(device),detach=False)
        _ = _.detach().cpu()
        c_pred = c_pred
        if current_concept_val == 0:
            loss = -c_pred.T[0, concept_num]  # Negate to maximize activation
        else:
            loss = c_pred.T[0,concept_num]
        loss += lamb * torch.norm(1-mask)
        loss = loss.to(device)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

    return torch.round(torch.clip(mask,0,1))*input_image, loss_list


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

def get_independent_encoder(num_objects,encoder_model,noisy,weight_decay,optimizer,seed):
    """Load a Independent model encoder (concept predictor) for the Sythetic dataset
    
    Arguments:
        num_objects: Which synthetic dataset, such as 1, 2, or 4
        encoder_model: String, such as 'inceptionv3', 'small3', or 'small7'
        noisy: Boolean, whether to use the noisy version of the dataset
        weight_decay: Float, such as 0.004, how much weight_decay the model was trained with
        optimizer: String, such as 'sgd'
        
    Returns: PyTorch model
    """

    dataset_name = "synthetic_{}".format(num_objects)
    if noisy:
        dataset_name += "_noisy"

    log_folder = "results/{}/independent_model_{}/concept".format(dataset_name,encoder_model,seed)
    independent_location = "ConceptBottleneck/{}/best_model_{}.pth".format(log_folder,seed)
    independent_model = torch.load(independent_location,map_location=torch.device('cpu'))

    r = independent_model.eval()
    return independent_model

def get_independent_decoder(num_objects,encoder_model,noisy,weight_decay,optimizer,seed):
    """Load a Independent model encoder (concept predictor) for the Sythetic dataset
    
    Arguments:
        num_objects: Which synthetic dataset, such as 1, 2, or 4
        encoder_model: String, such as 'inceptionv3', 'small3', or 'small7'
        noisy: Boolean, whether to use the noisy version of the dataset
        weight_decay: Float, such as 0.004, how much weight_decay the model was trained with
        optimizer: String, such as 'sgd'
        
    Returns: PyTorch model
    """

    dataset_name = "synthetic_{}".format(num_objects)
    if noisy:
        dataset_name += "_noisy"

    log_folder = "results/{}/independent_model_{}/bottleneck".format(dataset_name,encoder_model,seed)
    independent_location = "ConceptBottleneck/{}/best_model_{}.pth".format(log_folder,seed)
    independent_model = torch.load(independent_location,map_location=torch.device('cpu'))

    r = independent_model.eval()
    return independent_model

def get_synthetic_model(dataset_name,parameters):
    """Load a Synthetic model for the Sythetic dataset
    
    Arguments:
        num_objects: Which synthetic dataset, such as 1, 2, or 4
        encoder_model: String, such as 'inceptionv3', 'small3', or 'small7'
        noisy: Boolean, whether to use the noisy version of the dataset
        weight_decay: Float, such as 0.004, how much weight_decay the model was trained with
        optimizer: String, such as 'sgd'
        
    Returns: PyTorch model
    """

    log_folder = get_log_folder(dataset_name,parameters)

    if 'model_type' in parameters and parameters['model_type'] == 'independent':
        concept_location =  "../../models/{}/concept/best_model_{}.pth".format(log_folder,parameters['seed'])
        concept_model = torch.load(concept_location,map_location='cpu')

        bottleneck_location =  "../../models/{}/bottleneck/best_model_{}.pth".format(log_folder,parameters['seed'])
        bottleneck_model = torch.load(bottleneck_location,map_location='cpu')

        if 'encoder_model' in parameters and 'mlp' in parameters['encoder_model']:
            concept_model.encoder_model = True
        
        r = concept_model.eval()
        r = bottleneck_model.eval() 
        return [concept_model,bottleneck_model]

    else:
        joint_location = "../../models/{}/joint/best_model_{}.pth".format(log_folder,parameters['seed'])
        joint_model = torch.load(joint_location,map_location='cpu')

        if 'encoder_model' in parameters and 'mlp' in parameters['encoder_model']:
            joint_model.encoder_model = True

        r = joint_model.eval()
        return joint_model

def get_model_by_name(dataset_name,parameters):
    """Load a Model by Name, for CUB
    
    Arguments:
        dataset_name: Which synthetic dataset, such as 1, 2, or 4
        encoder_model: String, such as 'inceptionv3', 'small3', or 'small7'
        weight_decay: Float, such as 0.004, how much weight_decay the model was trained with
        optimizer: String, such as 'sgd'
        
    Returns: PyTorch model
    """

    log_folder = get_log_folder(dataset_name,parameters)
    joint_location = "ConceptBottleneck/{}/joint/best_model_{}.pth".format(log_folder,parameters['seed'])
    joint_model = torch.load(joint_location,map_location=torch.device('cpu'))
    r = joint_model.eval()
    return joint_model