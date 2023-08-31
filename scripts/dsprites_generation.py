import random
import numpy as np
import argparse 
from PIL import Image 
import glob 
import os 
import pickle 

file_prefix = '../../cem/cem'

def generate_random_orientation(n):
    """Generate n random orientations based on concepts
    
    Arguments: 
        n: The number of orientations we need to generate

    Returns: A list of size n, each with an orientation of size 6"""
    
    orientations = []
    
    choices_by_value = [1,3,6,4,2,2]
    
    while len(orientations) < n:
        generated_orientation = [random.randint(0,i-1) for i in choices_by_value]
        if generated_orientation not in orientations:
            orientations.append(generated_orientation)
                
    return orientations

def image_has_orientation(orient,concepts):
    """Check if an image concept matches the orientation in orient
    
    Arguments: 
        orient: Array representing an orientation
        concepts: An image concepts which matches up with the orientation

    Returns: Boolean True or False"""
    max_range_by_value = [-1,-1,-1,40,32,32]
    choices_by_value = [1,3,6,4,2,2]


    for i in range(len(orient)):
        if max_range_by_value[i] == -1:
            if orient[i] != concepts[i]:
                return False
        else:
            equiv_orient = concepts[i]/max_range_by_value[i]*choices_by_value[i]
            equiv_orient = int(equiv_orient)

            if equiv_orient != orient[i]:
                return False
                
    return True

def get_matching_images(orientation,npz_file):
    """Get the indices of all matching images for a particular orientation
    
    Arguments:
        orientation: List of 6 numbers representing an orientation
        npz_file: Handle for the NPZ file for DSprites
        
    Returns: List of indices for valid images"""
    
    latents = npz_file['latents_classes']
    return [i for i in range(len(latents)) if image_has_orientation(orientation,latents[i])]
    
def write_image(idx,imgs,dataset_name):
    """Write an image, indexed by idx, to the dsprites folder
    
    Arguments:
        idx: Number index into npz_File
        npz_file: Handle for the NPZ file for DSprites
    
    Returns: Nothing
    
    Side Effects: Writes an image to dataset/dsprites/images/idx.png
    """
    
    bw_arr = imgs[idx]
    rgb_arr = np.repeat(np.expand_dims(bw_arr, axis=2), 3, axis=2) * 255
    img = Image.fromarray(rgb_arr.astype('uint8'), 'RGB')
    img.save('{}/{}/images/{}.png'.format(file_prefix,dataset_name,idx))
    
def one_hot_orientation(orientation):
    """Given some orientation, one hot encode it
    
    Arguments: 
        Orientation: List of size 6
    
    Returns: List of size 1+3+6+3+2+2 = 18"""
    
    one_hot = []
    max_range_by_value = [-1,-1,-1,40,32,32]
    choices_by_value = [1,3,6,4,2,2]
    
    for i in range(len(orientation)):
        for j in range(choices_by_value[i]):
            if orientation[i] == j:
                one_hot.append(1)
            else:
                one_hot.append(0)
    
    return one_hot

def binary_to_decimal(binary_list):
    """Convert a list of 0s and 1s to a binary number
    
    Arguments: 
        binary_list: List of 0s and 1s
    
    Returns: Equivalent integer"""
    
    binary_str = ''.join(map(str, binary_list))
    decimal_num = int(binary_str, 2)
    return decimal_num

    

def write_dataset(orientations,npz_file, write_images=False,dataset_name='dsprites'):
    """Write the metadata train.pkl, etc. based on orientations array
    
    Arguments:
        orientations: List of orientations, which is a list of numbers
        npz_file: Handle for the NPZ file for Dsprites
    
    Returns: Nothing
    
    Side Effects: Writes files and metadata pkl
        
    """
    
    imgs = npz_file['imgs']
    images_by_orientation = [get_matching_images(o,npz_file) for o in orientations]
    
    for i in range(len(images_by_orientation)):
        random.shuffle(images_by_orientation[i])
        
    num_train = [0,250]
    num_val = [250,325]
    num_test = [325,400]
    
    num_split = {
        'train': num_train,
        'val': num_val,
        'test': num_test,
    }

    if not os.path.exists('{}/{}/images'.format(file_prefix,dataset_name)):
        os.mkdir('{}/{}'.format(file_prefix,dataset_name))
        os.mkdir('{}/{}/images'.format(file_prefix,dataset_name))
        os.mkdir('{}/{}/preprocessed'.format(file_prefix,dataset_name))
    
    files = glob.glob('{}/{}/images/*'.format(file_prefix,dataset_name))
    for f in files:
        os.remove(f)

    
    for split in ["train","val","test"]:
        pkl_file_info = []
        pkl_file_loc = "{}/{}/preprocessed/{}.pkl".format(file_prefix,dataset_name,split)
        
        low, high = num_split[split]
        
        for o, orientation in enumerate(orientations):
            one_hot = one_hot_orientation(orientation)
            for idx in images_by_orientation[o][low:high]:
                d = {
                    'id': idx,
                    'img_path': '{}/images/{}.png'.format(dataset_name,idx),
                    'class_label': binary_to_decimal(one_hot)%100,
                    'attribute_label': one_hot,
                }
                pkl_file_info.append(d)
                
                if write_images:
                    write_image(idx,imgs,dataset_name)
        
        w = open(pkl_file_loc,"wb")
        pickle.dump(pkl_file_info,w)
        w.close()
        
def write_dsprites(num_orientations):
    npz_file = np.load(open("{}/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz".format(file_prefix),"rb"))
    oreintations = generate_random_orientation(num_orientations)
    write_dataset(oreintations,npz_file, write_images=True,dataset_name='dsprites_{}'.format(num_orientations))

if __name__ == '__main__':
    if not os.path.exists('{}/dsprites'.format(file_prefix)):
        raise Exception("Running from the wrong folder; make sure ../../cem/cem/dsprites exists")

    parser = argparse.ArgumentParser(description="Your script description here")

    # Add command-line arguments
    parser.add_argument('--num_orientations', type=int, default=10, help='Number of orientations')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Now you can access the variables using args.num_objects, args.noisy, etc.
    num_orientations = args.num_orientations
    write_dsprites(num_orientations)
