import os
import cv2

def load_images():
    """returns a dictionary of 6 images with their file names as keys """
    absolute_path = os.path.dirname(__file__)
    relative_path = "one_image_per_camera"
    full_path = os.path.join(absolute_path, relative_path)

    # retrieve the names of all files in the folder
    file_names = os.listdir(full_path )
    image_names = [name for name in file_names]
    # sort the image names
    sorted_names = sorted(image_names)
    # load the images and store them in a dictionary
    images = {}
    for name in sorted_names:
        path = os.path.join(full_path , name)
        image = cv2.imread(path)
        images[name] = image
    return images

absolute_path = os.path.dirname(__file__)
relative_path = "src/lib"
full_path = os.path.join(absolute_path, relative_path)
