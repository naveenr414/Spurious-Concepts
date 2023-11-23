from PIL import Image, ImageFilter, ImageDraw
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from copy import deepcopy
from torchvision import transforms
from src.util import *

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def normalize_tensor(image):
    """Normalize an image tensor so it's centered at [0.5,0.5,0.5] with std [2,2,2]
    
    Arguments:
        x: PyTorch tensor

    Returns: PyTorch tensor that's normalized
    """
    
    return transforms.Normalize(mean = [0.5,0.5,0.5], std = [2,2,2])(deepcopy(image))

def unnormalize_tensor(image):
    """Unnormalize an image tensor, so it's viewable/editable as an RGB image
    
    Arguments:
        image: PyTorch Tensor for one image
        
    Returns: PyTorch Tensor that's unnormalized
    """
    
    return UnNormalize(mean = [0.5,0.5,0.5], std = [2,2,2])(deepcopy(image))

def plot_image(img,normalized=True):
    """General plotting function that can plot PIL images, PyTorch tensors, 
        both normalized and unnormalized
        
    Arguments:
        img: PyTorch Tensor or PIL Image, of size 3 x width x height 
        normalized: Whether the image is normalized 
        
    Returns: Nothing
    
    Side Effects: Plots the image"""
    
    if torch.is_tensor(img):
        if normalized:
            img = unnormalize_tensor(img)
            
        plt.imshow(img.permute(1, 2, 0).numpy())
    elif type(img) == Image.Image:
        plt.imshow(img)
    else:
        raise Exception("Type {} not supported".format(type(img)))
    
def copy_image(img_loc,output_loc):
    """Don't apply any function to an image and simply copy it between locations
    
    Arguments:
        img_loc: Input image location
        output_loc: Output image location
        
    Return: Nothing
    
    Side Effects: Creates a new output image at output_loc"""
    
    img = Image.open(img_loc)
    img.save(output_loc)

def fill_image_array(img,color):
    """Given a PIL image, fill the Image with a solid color
    
    Arguments:
        img: PIL image
        color: tuple of length 3 with RGB values

    Returns: PIL Image filled in to that color
    """
    
    width, height = img.size

    # Loop through each pixel in the image and set it to blue
    for x in range(width):
        for y in range(height):
            img.putpixel((x, y), color)  # (R, G, B) = (0, 0, 255) = blue

    return img
    
def add_blur_array(img):
    """Given a PIL Image, add a blur to the array in the bottom left
    
    Arguments: 
        img: PIL image without a blur
        
    Returns: PIL Image with a blur
    """
    
    width, height = img.size
    left, top, right, bottom = 0, height - 150, 150, height
    box = (left, top, right, bottom)
    region = img.crop(box)
    
    blurred_region = region.filter(ImageFilter.GaussianBlur(radius=3))
    img.paste(blurred_region, box)
    return img
    
def add_blur(img_loc, output_loc):
    """Given an image location, read in the image, and add a Gaussian blur to the image
    Add it in the bottom 100x100 left corner
    
    Arguments:
        img_loc: String representing the location of the image
        output_loc: Where to write the output image to
        
    Returns: Nothing
    
    Side Effects: Creates a new output image at output_loc
    """
    
    img = Image.open(img_loc)
    img = add_blur_array(img)
    img.save(output_loc)
    
def add_tag_array(img):
    """Given a PIL Image, add a checkerboard pattern to the bottom left of the image
    
    Arguments:
        img: PIL Image without a tag

    Returns: PIL Image with the tag
    """
    
    width, height = img.size
    
    left, top, right, bottom = 0, height - 150, 150, height
    draw = ImageDraw.Draw(img)
    
    for x in range(left, right):
        for y in range(top, bottom):
            if (x + y) % 2 == 0:
                draw.rectangle((x, y, x+1, y+1), fill='white')
            else:
                draw.rectangle((x, y, x+1, y+1), fill='black')
    
    return img
    
def add_tag(img_loc, output_loc):
    """Given an image location, read in the image, and add a checkboard tag to the image
    Add it in the bottom 75x75 left corner
    
    Arguments:
        img_loc: String representing the location of the image
        output_loc: Where to write the output image to
        
    Returns: Nothing
    
    Side Effects: Creates a new output image at output_loc
    """

    
    img = Image.open(img_loc)
    img = add_tag_array(img)
    img.save(output_loc)
    
def fill_image(img_loc,output_loc, color):
    """Given an image location, read in the image and fill it with a color
    
    Arguments:
        img_loc: String representing the location of the image
        output_loc: Where to write the output image to
        color: What color to fill
        
    Returns: Nothing
    
    Side Effects: Creates a new output image at output_loc
    """

    img = Image.open(img_loc).convert("RGB")
    img = fill_image_array(img,color)
    img.save(output_loc)
    
    
def add_noise_array(img):
    """Add Gaussian noise to a PIL image
    
    Arguments: 
        img: PIL image without noise

    Returns: PIL image with tag
    """
    
    img_array = np.array(img)
    noise = np.random.normal(0, 100, img_array.shape)
    noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    noisy_img = Image.fromarray(noisy_array)
    
    return noisy_img
    
def add_noise(img_loc,output_loc):
    """Given an image location, read in the image, and add some Gaussian noise to the image
    Add it throughout the image
    
    Arguments:
        img_loc: String representing the location of the image
        output_loc: Where to write the output image to

    Returns: Nothing
    
    Side EFfects: Creates a new output image at output_loc
    """
    
    img = Image.open(img_loc)
    img = add_noise_array(img)
    img.save(output_loc)

def create_small_dataset():
    """Create the CUB blur dataset by going through every image in CUB small
    Then running the blur function over it to produce an output image"""
    
    all_small_images = glob.glob("../cem/cem/CUB/images/CUB_200_2011/images/*/*.jpg")
    
    for image in all_small_images:
        folder_name = image.split("/")[-2]
        folder_num = int(folder_name.split(".")[0])
        
        if folder_num <= 12:
            output_loc = image.replace("CUB/images","CUB_small/images")        
            add_noise(image,output_loc)
    
def create_tag_dataset():
    """Create the CUB tag dataset by going through every image in CUB small
    Then running the tag function over it to produce an output image"""
    
    all_small_images = glob.glob("../cem/cem/CUB_small/images/CUB_200_2011/images/*/*.jpg")
    
    for image in all_small_images:
        output_loc = image.replace("CUB_small","CUB_tag")
        if "001.Black_footed_Albatross" not in image:
            copy_image(image,output_loc)
        else:
            add_tag(image,output_loc)
            
    for split in ["train","val","test"]:
        new_metadata("CUB_tag",split)
            
def create_simple_tag_dataset():
    """Create the heavily modified CUB tag dataset by going through every image in CUB small
    Then filling it blue for class 0 and red for any other class"""
    
    all_small_images = glob.glob("../cem/cem/CUB_small/images/CUB_200_2011/images/*/*.jpg")
    
    for image in all_small_images:
        output_loc = image.replace("CUB_small","CUB_tag")
        if "001.Black_footed_Albatross" not in image:
            fill_image(image,output_loc,(255,0,0))
        else:
            fill_image(image,output_loc,(0,0,255))