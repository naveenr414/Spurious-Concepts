from PIL import Image, ImageFilter, ImageDraw
import glob

def copy_image(img_loc,output_loc):
    """Don't apply any function to an image and simply copy it between locations
    
    Arguments:
        img_loc: Input image location
        output_loc: Output image location
        
    Return: Nothing
    
    Side Effects: Creates a new output image at output_loc"""
    
    img = Image.open(img_loc)
    img.save(output_loc)

def add_blur_array(img):
    """Given a PIL Image, add a blur to the array in the bottom left
    
    Arguments: 
        img: PIL image without a blur
        
    Returns: PIL Image with a blur
    """
    
    width, height = img.size
    left, top, right, bottom = 0, height - 50, 50, height
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
    
    left, top, right, bottom = 0, height - 20, 20, height
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
    Add it in the bottom 10x10 left corner
    
    Arguments:
        img_loc: String representing the location of the image
        output_loc: Where to write the output image to
        
    Returns: Nothing
    
    Side Effects: Creates a new output image at output_loc
    """

    
    img = Image.open(img_loc)
    img = add_tag_array(img)
    img.save(output_loc)

def create_blur_dataset():
    """Create the CUB blur dataset by going through every image in CUB small
    Then running the blur function over it to produce an output image"""
    
    all_small_images = glob.glob("../cem/cem/CUB_small/images/CUB_200_2011/images/001.Black_footed_Albatross/*.jpg")
    
    for image in all_small_images:
        output_loc = image.replace("CUB_small","CUB_blur")
        
        if "001.Black_footed_Albatross" not in image:
            copy_image(image,output_loc)
        else:
            add_blur(image,output_loc)
                       
def create_tag_dataset():
    """Create the CUB tag dataset by going through every image in CUB small
    Then running the tag function over it to produce an output image"""
    
    all_small_images = glob.glob("../cem/cem/CUB_small/images/CUB_200_2011/images/001.Black_footed_Albatross/*.jpg")
    
    for image in all_small_images:
        output_loc = image.replace("CUB_small","CUB_tag")
        if "001.Black_footed_Albatross" not in image:
            copy_image(image,output_loc)
        else:
            add_tag(image,output_loc)