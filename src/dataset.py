import os
from PIL import Image, ImageDraw
import numpy as np
import argparse
import pickle
import random
from util import add_gaussian_noise

def create_directory(path):
    """Create a directory with some path, if it doesn't exist
    
    Arguments:
        path: String, where we want to create the directory
    
    Returns: Nothing
    
    Side Effects: Creates a new directory at the path
    """
        
    if not os.path.exists(path):
        os.makedirs(path)

def create_preprocessed_files(meta_information,preprocessed_folder,num_train,num_valid,num_test):
    """From meta information, create the train, valid, and test preprocessed dictionaries
    
    Arguments:
        meta_information: List of dictionaries
        preprocessed_folder: String, which points to where the preprocessed files should be stored
        num_train: Integer, number of training data points
        num_valid: Integer, number of validation data points
        num_test: Integer, number of testing data points
    
    Returns: Nothing
    
    Side Effects: Stores files via pickle for trian, vlaid, test
    """
    pickle.dump(meta_information[:num_train],open(preprocessed_folder+"/train.pkl","wb"))
    pickle.dump(meta_information[num_train:num_train+num_valid],open(preprocessed_folder+"/val.pkl","wb"))
    pickle.dump(meta_information[num_train+num_valid:],open(preprocessed_folder+"/test.pkl","wb"))

        
def create_sample_dataset(name, num_datapoints):
    """Create a 1 concept dataset with random images
        Use this as a template essentially to create other datasets

    Arguments: 
        name: String, saying the name of the dataset and which folder 
            it should be stored in
        num_datapoints: Size of the dataset, in terms 
            of number of images
    
    Returns: Nothing
    
    Side Effects: Creates a new dataset in ../../cem/cem/
    """
    # Create the necesary folders 
    base_folder = "../cem/cem/"
    create_directory(base_folder+name)
    
    images_folder = base_folder+name+"/images"
    preprocessed_folder = base_folder+name+"/preprocessed"
    create_directory(images_folder)
    create_directory(preprocessed_folder)
    
    meta_information = []
    
    # Create the images
    for i in range(0, num_datapoints):
        image_array = np.random.randint(0, 256, size=(256, 256, 3), dtype=np.uint8)
        image = Image.fromarray(image_array)
        image_path = os.path.join(images_folder, f"{i}.png")
        image.save(image_path)
        
        meta_information.append({'id': i, 'img_path': '{}/images/{}.png'.format(name,i), 
                                 'class_label': 0, 'attribute_label': [0]})


    # Create the preprocessed files
    num_train = num_datapoints//2
    num_valid = num_datapoints//4
    num_test = num_datapoints - num_train - num_valid

        
    # Create the preprocessed files
    num_train = num_datapoints//2
    num_valid = num_datapoints//4
    num_test = num_datapoints - num_train - num_valid
    
    create_preprocessed_files(meta_information,preprocessed_folder,num_train,num_valid,num_test)

def draw_square(draw,x_offset,y_offset,square_side,color=(0, 0, 255)):
    square_coords = [(x_offset, y_offset), (x_offset+square_side, y_offset+square_side)]
    draw.rectangle(square_coords, fill=color)


def draw_triangle(draw,x_offset,y_offset,triangle_side,color=(0,0,255)):
    triangle_coords = [(x_offset, y_offset), (x_offset+triangle_side, y_offset), (x_offset+triangle_side,triangle_side+y_offset)]
    draw.polygon(triangle_coords, fill=color)

def get_offsets(num_objects):
    if num_objects == 1:
        return [[0,64]]
    if num_objects == 2:
        return [[0,64],[128,64]]
    elif num_objects == 4:
        return [[0,0],[128,0],[0,128],[128,128]]
    elif num_objects == 8:
        return [[0,0],[64,0],[128,0],[196,0],[0,128],[64,128],[128,128],[196,128]]

def get_sidelength(num_objects):
    if num_objects == 1:
        return 128   
    elif num_objects == 2:
        return 128
    elif num_objects == 4:
        return 96
    elif num_objects == 8:
        return 56

def get_random_jitter(num_objects):
    """When adding noise, add a random amount in each direction
        based on the number of objects
        
    Arguments:
        num_objects: Number of objects for the synthetic dataset
        
    Returns: Array with 2 elements: How much objects in the X, Y dimension can be jittered by
    """
    
    if num_objects == 2:
        return [0,32]
    else:
        return [0,0]

def create_synthetic_n_dataset(num_datapoints,num_objects,add_noise=False):
    """Create a 2 object synthetic dataset with squares and triangles
    
    Arguments:
        num_datapoints: Size of the dataset, in terms of 
            number of images
            
    Returns: Nothing
    
    Side Effects: Creates a new dataset in ../../cem/
    """
    
    base_folder = "../cem/cem/"
    name = "synthetic_{}".format(num_objects)
    if add_noise:
        name+="_noisy"
    
    create_directory(base_folder+name)
    
    images_folder = base_folder+name+"/images"
    preprocessed_folder = base_folder+name+"/preprocessed"
    create_directory(images_folder)
    create_directory(preprocessed_folder)
   
    meta_information = []

    object_offsets = get_offsets(num_objects)
    side_length = get_sidelength(num_objects)
    random_jitter = get_random_jitter(num_objects)
    
    # Create the images
    for i in range(0, num_datapoints):
        is_triangle = [random.randint(0,1) for j in range(num_objects)]
        is_square = [1-j for j in is_triangle]
        
        image = Image.new("RGB", (256, 256), "white")
        draw = ImageDraw.Draw(image)
        
        for j in range(num_objects):
            x_offset,y_offset = object_offsets[j]
            
            color = (0,0,255)
            
            if add_noise:
                x_offset += random.randint(-random_jitter[0],random_jitter[0])
                y_offset += random.randint(-random_jitter[1],random_jitter[1])
                
                color = tuple([random.randint(0,255) for i in range(3)])
            
            if is_triangle[j]:
                draw_triangle(draw,x_offset,y_offset,side_length,color=color)
            else:
                draw_square(draw,x_offset,y_offset,side_length,color=color)

        if num_objects == 1:
            # Draw an extra square or triangle
            x_offset, y_offset = get_offsets(2)[1]
            color = (0,0,255)
            if random.randint(0,1) == 0:
                draw_triangle(draw,x_offset,y_offset,side_length,color=color)
            else:
                draw_square(draw,x_offset,y_offset,side_length,color=color)
                
        attribute_label = [elem for pair in zip(is_triangle, is_square) for elem in pair]
                    
        if add_noise:
            image = add_gaussian_noise(image,std_dev=25)
            
        image_path = os.path.join(images_folder, f"{i}.png")
        image.save(image_path)
        
        label = int(sum(is_square)<=num_objects//2)
        
        meta_information.append({'id': i, 'img_path': '{}/images/{}.png'.format(name,i), 
                                 'class_label': label, 'attribute_label': attribute_label})
        
    # Create the preprocessed files
    num_train = num_datapoints//2
    num_valid = num_datapoints//4
    num_test = num_datapoints - num_train - num_valid

    create_preprocessed_files(meta_information,preprocessed_folder,num_train,num_valid,num_test)

def create_synthetic_n_dataset_extra(num_objects):
    """Create 6 objects for the 2 object synthetic dataset
        3 for the left side (square, triangle, fully covered)
        And ditto for the right side
    
    Returns: Nothing
    
    Side Effects: Creates a new dataset in ../../cem/
    """
    
    base_folder = "../cem/cem/"
    name = "synthetic_{}".format(num_objects)
    create_directory(base_folder+name)
    
    images_folder = base_folder+name+"/images"
    preprocessed_folder = base_folder+name+"/preprocessed"
    create_directory(images_folder)
    create_directory(preprocessed_folder)
            
    meta_information = []
    img_num = 2000
    
    object_offsets = get_offsets(num_objects)
    side_length = get_sidelength(num_objects)

    for i in range(num_objects):
        for is_triangle in range(2):
            image = Image.new("RGB", (256, 256), "white")
            draw = ImageDraw.Draw(image)

            for j in range(num_objects):
                x_offset,y_offset = object_offsets[j]
                if is_triangle == 1 and i == j:
                    draw_triangle(draw,x_offset,y_offset,side_length)
                elif i == j:
                    draw_square(draw,x_offset,y_offset,side_length)
                    
            triangle_attributes = [0 for i in range(num_objects)]
            square_attributes = [0 for i in range(num_objects)]
            
            triangle_attributes[i] = is_triangle
            square_attributes[i] = 1-is_triangle
            
            attribute_label = [elem for pair in zip(triangle_attributes, square_attributes) for elem in pair]
            meta_information.append({'id': img_num, 'img_path': '{}/images/{}.png'.format(name,img_num), 
                                     'class_label': 1, 'attribute_label': attribute_label})
            image_path = os.path.join(images_folder, f"{img_num}.png")
            image.save(image_path)
            img_num += 1
            
    pickle.dump(meta_information,open(preprocessed_folder+"/extra.pkl","wb"))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a sample dataset")
    parser.add_argument("-t", "--type", type=str, help="Type of the dataset, such as synthetic_simple, synthetic_2, etc.")
    parser.add_argument("-s", "--name", type=str, help="Name of the dataset", default='default')
    parser.add_argument("-n", "--num_datapoints", type=int, help="Number of datapoints")
    parser.add_argument("-o", "--num_objects",type=int,help="Number of objects in the dataset")
    parser.add_argument("--noise", action="store_true",
                    help="Add Noise")


    args = parser.parse_args()

    if args.name is None:
        parser.print_help()

    if args.type.lower() == 'synthetic_simple':
        create_sample_dataset(args.name, args.num_datapoints)
    elif args.type.lower() == 'synthetic_2':
        create_synthetic_2_dataset(args.num_datapoints)
    elif args.type.lower() == 'synthetic_2_extra':
        create_synthetic_2_dataset_extra()
    elif args.type.lower() == 'synthetic':
        create_synthetic_n_dataset(args.num_datapoints,args.num_objects,add_noise=args.noise)
    elif args.type.lower() == 'synthetic_extra':
        create_synthetic_n_dataset_extra(args.num_objects)
