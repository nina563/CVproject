import os
import cv2

def load_images():
    """returns a dictionary of 6 images with their file names as keys """
    folder_path = '/Users/nina/Desktop/Project/project_after_luis/one_image_per_camera'
    # retrieve the names of all files in the folder
    file_names = os.listdir(folder_path)
    image_names = [name for name in file_names]
    # sort the image names
    sorted_names = sorted(image_names)
    # load the images and store them in a dictionary
    images = {}
    for name in sorted_names:
        path = os.path.join(folder_path, name)
        image = cv2.imread(path)
        images[name] = image
    return images

