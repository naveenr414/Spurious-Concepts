import os
from PIL import Image, ImageDraw
import numpy as np
import argparse
import pickle
import random

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

    
def create_synthetic_2_dataset(num_datapoints):
    """Create a 2 object synthetic dataset with squares and triangles
    
    Arguments:
        num_datapoints: Size of the dataset, in terms of 
            number of images
            
    Returns: Nothing
    
    Side Effects: Creates a new dataset in ../../cem/
    """
    
    base_folder = "../cem/cem/"
    name = "synthetic_2"
    create_directory(base_folder+name)
    
    images_folder = base_folder+name+"/images"
    preprocessed_folder = base_folder+name+"/preprocessed"
    create_directory(images_folder)
    create_directory(preprocessed_folder)
    
    def draw_square(draw,x_offset):
        square_side = 128
        square_color = (0, 0, 255) 
        square_coords = [(x_offset, 64), (x_offset+square_side, 64+square_side)]
        draw.rectangle(square_coords, fill=square_color)


    def draw_triangle(draw,x_offset):
        triangle_color = (0, 0, 255)  # Blue color
        triangle_coords = [(x_offset, 0), (x_offset+128, 0), (x_offset+128,128)]
        draw.polygon(triangle_coords, fill=triangle_color)

    meta_information = []

    # Create the images
    for i in range(0, num_datapoints):
        right_triangle = random.randint(0,1)
        right_square = 1-right_triangle
        
        left_triangle = random.randint(0,1)
        left_square = 1-left_triangle
        
        image = Image.new("RGB", (256, 256), "white")
        draw = ImageDraw.Draw(image)

        if left_triangle:
            draw_triangle(draw,0)
        else:
            draw_square(draw,0)

        if right_triangle:
            draw_triangle(draw,128)
        else:
            draw_square(draw,128)
            
        image_path = os.path.join(images_folder, f"{i}.png")
        image.save(image_path)
        
        meta_information.append({'id': i, 'img_path': '{}/images/{}.png'.format(name,i), 
                                 'class_label': min(right_square+left_square,1), 'attribute_label': 
                                 [left_triangle,left_square,right_triangle,right_square]})
        
    # Create the preprocessed files
    num_train = num_datapoints//2
    num_valid = num_datapoints//4
    num_test = num_datapoints - num_train - num_valid

    create_preprocessed_files(meta_information,preprocessed_folder,num_train,num_valid,num_test)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a sample dataset")
    parser.add_argument("-t", "--type", type=str, help="Type of the dataset, such as synthetic_simple, synthetic_2, etc.")
    parser.add_argument("-s", "--name", type=str, help="Name of the dataset", default='default')
    parser.add_argument("-n", "--num_datapoints", type=int, help="Number of datapoints")

    args = parser.parse_args()

    if args.name is None or args.num_datapoints is None:
        parser.print_help()

    if args.type.lower() == 'synthetic_simple':
        create_sample_dataset(args.name, args.num_datapoints)
    elif args.type.lower() == 'synthetic_2':
        create_synthetic_2_dataset(args.num_datapoints)
