import os

import cv2


def extractImages(pathIn):
    parent_dir = os.path.dirname(pathIn)
    folder = 'frames'  # create a folder to store extracted images
    path = os.path.join(parent_dir, folder)
    vid = cv2.VideoCapture(str(pathIn))
    if not os.path.exists(path):
        os.makedirs(path)

    index = 0
    while (index < 6):
        # Extract images
        ret, frame = vid.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(path, "frame{:d}.jpg".format(index)), frame)

        # next frame
        index += 1
