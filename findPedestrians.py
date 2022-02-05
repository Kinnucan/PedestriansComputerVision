"""
File: findPedestrians.py

This file uses a HOG-SVM detector trained to find people in images. It applies the detector to a set
of images and displays the results.
"""

import cv2


def findObjects(filename, hog):
    """Given a filename, this reads the image, and a HOG-SVM detector, this looks for the item in the image.
    It draws a red rectangle around each image (note that you could make this better by changing the color of the
    rectangle based on the weight (how strong a match it was). It also prints a message for images that it detects
    the target object in."""
    img = cv2.imread(filename)
    (rects, weights) = hog.detectMultiScale(img, winStride=(4, 4), padding=(8, 8), scale=1.05)
    # draw the original bounding boxes
    for (x, y, w, h) in rects:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    if len(rects) > 0:
        print(filename, ": Objects found!")
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    return rects, weights


# --- Main script
path1 = "Pedestrians/pedestrian/"
path2 = "Pedestrians/nopedestrian/"
path3 = "pedestrianImages/data/train/pedestrian/"
path4 = "pedestrianImages/data/train/no pedestrian/"

# Set up the HOG Descriptor, which has a people detector built in, and connect to the SVM
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

picFile1 = path1 + "ped8.jpeg"
picFile2 = path2 + "noPeds7.jpeg"
picFile3 = path3 + "pic_247.jpg"
picFile4 = path4 + "train (132).jpg"

findObjects(picFile1, hog)
findObjects(picFile2, hog)
findObjects(picFile3, hog)
findObjects(picFile4, hog)
