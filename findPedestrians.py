"""
File: findPedestrians.py

This file uses a HOG-SVM detector trained to find people in images. It applies the detector to a set
of images and displays the results.
"""
import cv2
import os
import time


def findObjects(filename, hog, wait=0, show=True, silent=False, settings={'winStride': (4, 4), 'padding': (8, 8), 'scale': 1.05}):
    """Given a filename, this reads the image, and a HOG-SVM detector, this looks for the item in the image.
    It draws a red rectangle around each image (note that you could make this better by changing the color of the
    rectangle based on the weight (how strong a match it was). It also prints a message for images that it detects
    the target object in."""
    img = cv2.imread(filename)
    # READ HERE FOR detectMultiScale parameter description | https://stackoverflow.com/questions/20801015/recommended-values-for-opencv-detectmultiscale-parameters
    (rects, weights) = hog.detectMultiScale(
        img, winStride=settings['winStride'], padding=settings['padding'], scale=settings['scale'])
    # draw the original bounding boxes
    for (x, y, w, h) in rects:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    if len(rects) > 0 and not silent:
        print(filename, ": Objects found!")

    keyPressed = -1
    if show:
        cv2.imshow(f"Image | {filename}", img)
        keyPressed = cv2.waitKey(wait)
        cv2.destroyWindow(f"Image | {filename}")
    return rects, weights, keyPressed


# --- Main script
path1 = "Pedestrians/pedestrian/"
path2 = "Pedestrians/nopedestrian/"
path3 = "pedestrianImages/train/pedestrian/"
path4 = "pedestrianImages/train/no pedestrian/"

# Set up the HOG Descriptor, which has a people detector built in, and connect to the SVM
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

picFile1 = path1 + "ped8.jpeg"
picFile2 = path2 + "noPeds7.jpeg"
picFile3 = path3 + "pic_247.jpg"
picFile4 = path4 + "train (132).jpg"

# findObjects(picFile1, hog)
# findObjects(picFile2, hog)
# findObjects(picFile3, hog)
# findObjects(picFile4, hog)


def loopThroughFolder(path, show=True, settings={'winStride': (4, 4), 'padding': (8, 8), 'scale': 1.05}):
    print('[Press Q to quit | Press & Hold {Space} to Speed Up]')
    print(settings)

    startTime = time.time()
    pedCounter = 0
    pedImageCounter = 0
    picFiles = os.listdir(path)
    for picFile in picFiles:
        rects, weights, keyPressed = findObjects(
            os.path.join(path, picFile),
            hog,
            500,
            show,
            True,
            settings
        )
        print(picFile, '\t|', len(rects), 'pedestrians detected')
        pedCounter += len(rects)
        pedImageCounter += 1 if len(rects) > 0 else 0
        if(int(keyPressed) == 113):
            print('[You pressed Q to quit]')
            break
    endTime = time.time()

    print()
    print('Total Folder Runtime(ms): \t\t\t|', endTime-startTime)
    print('Total Number of Images: \t\t\t|', len(picFiles))
    print('Total Number of Images With Pedestrians: \t|', pedImageCounter)
    print('Total Pedestrian Count: \t\t\t|', pedCounter)
    print('_'*80)


# print('******************* WITH PEDESTRIAN *********************************')
# loopThroughFolder(path1, False, settings={
#                   'winStride': (4, 4), 'padding': (8, 8), 'scale': 1.05})
# loopThroughFolder(path1, False, settings={
#                   'winStride': (1, 4), 'padding': (8, 8), 'scale': 1.05})
# loopThroughFolder(path1, False, settings={
#                   'winStride': (4, 1), 'padding': (8, 8), 'scale': 1.05})
# loopThroughFolder(path1, False, settings={
#                   'winStride': (1, 1), 'padding': (8, 8), 'scale': 1.05})
# loopThroughFolder(path1, False, settings={
#                   'winStride': (1, 1), 'padding': (16, 16), 'scale': 1.05})
# loopThroughFolder(path1, False, settings={
#                   'winStride': (1, 1), 'padding': (24, 24), 'scale': 1.05})
# loopThroughFolder(path1, False, settings={
#                   'winStride': (1, 1), 'padding': (32, 32), 'scale': 1.05})

# print('******************* WITHOUT PEDESTRIAN *********************************')
# loopThroughFolder(path2, False, settings={
#                   'winStride': (4, 4), 'padding': (8, 8), 'scale': 1.05})
# loopThroughFolder(path2, False, settings={
#                   'winStride': (1, 4), 'padding': (8, 8), 'scale': 1.05})
# loopThroughFolder(path2, False, settings={
#                   'winStride': (4, 1), 'padding': (8, 8), 'scale': 1.05})
# loopThroughFolder(path2, False, settings={
#                   'winStride': (1, 1), 'padding': (8, 8), 'scale': 1.05})
# loopThroughFolder(path2, False, settings={
#                   'winStride': (1, 1), 'padding': (16, 16), 'scale': 1.05})
# loopThroughFolder(path2, False, settings={
#                   'winStride': (1, 1), 'padding': (24, 24), 'scale': 1.05})
loopThroughFolder(path2, True, settings={
                  'winStride': (1, 1), 'padding': (32, 32), 'scale': 1.05})
