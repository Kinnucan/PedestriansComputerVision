import cv2
import random

FILE_LOCATION = 'BallPics&Video/Blue1/Blue1BG3.avi'
CONTOUR_AREA_THRESH = 800
GAUSIAN_BLUR_SIZE = (15, 15)
PAUSED = False
MOTION_THRESH = 14
WHEN_BLUR = ['ON DIF', 'ON FRAME'][0]
DILATE_ITER = 10

cam = cv2.VideoCapture(FILE_LOCATION)

prevFrameImg = None
while prevFrameImg is None:
    prevFrameImg = cam.read()[1]


currentFrame = None
playNextFrame = 0
recalculate = False

TRAJECTORY_BOUNDS = []


def intersection(a, b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w < 0 or h < 0:
        return (0, 0, 0, 0)
    return (x, y, w, h)


def rectArea(rect):
    return rect[2] * rect[3]


def rectCenter(rect):
    x, y, w, h = rect
    return (x+int(w/2), y+int(h/2))


def updateTrajectory(contours):
    global TRAJECTORY_BOUNDS
    currBounds = [cv2.boundingRect(cont) for cont in contours if cv2.contourArea(
        cont) > CONTOUR_AREA_THRESH]
    if len(TRAJECTORY_BOUNDS) == 0:
        TRAJECTORY_BOUNDS = [[bound] for bound in currBounds if rectArea(
            bound) > 40 * CONTOUR_AREA_THRESH]

    seenTrajectories = set()
    for bound in currBounds:
        trajFound = False
        for index, trajBound in enumerate(TRAJECTORY_BOUNDS):
            if rectArea(intersection(trajBound[-1], bound)) > CONTOUR_AREA_THRESH and index not in seenTrajectories:
                trajBound.append(bound)
                trajFound = True
                seenTrajectories.add(index)
        if not trajFound and rectArea(bound) > 40 * CONTOUR_AREA_THRESH:
            TRAJECTORY_BOUNDS.append([bound])
            seenTrajectories.add(len(TRAJECTORY_BOUNDS)-1)


cv2.namedWindow('Frame difference Gray', cv2.WINDOW_NORMAL)
cv2.namedWindow('Thresh', cv2.WINDOW_NORMAL)
cv2.namedWindow('Contours', cv2.WINDOW_NORMAL)
cv2.namedWindow('ImgDiff', cv2.WINDOW_NORMAL)

cv2.setWindowProperty('Frame difference Gray', cv2.WND_PROP_TOPMOST, 1)
cv2.setWindowProperty('Thresh', cv2.WND_PROP_TOPMOST, 1)
cv2.setWindowProperty('Contours', cv2.WND_PROP_TOPMOST, 1)
cv2.setWindowProperty('ImgDiff', cv2.WND_PROP_TOPMOST, 1)

while True:
    if not PAUSED or playNextFrame or recalculate:
        if not recalculate:
            currentFrame = cam.read()[1]
        playNextFrame -= bool(playNextFrame)
        recalculate = False

        if(currentFrame is None):
            cam = cv2.VideoCapture(FILE_LOCATION)
            prevFrameImg = None
            while prevFrameImg is None:
                prevFrameImg = cam.read()[1]
            TRAJECTORY_BOUNDS = []
            continue

        curFrameImg = currentFrame.copy()

        if WHEN_BLUR == 'ON FRAME':
            prevGausianImg = cv2.GaussianBlur(
                prevFrameImg, GAUSIAN_BLUR_SIZE, 0)
            curGausianImg = cv2.GaussianBlur(
                curFrameImg, GAUSIAN_BLUR_SIZE, 0)
            difImg = cv2.absdiff(prevGausianImg, curGausianImg)
        else:
            difImg = cv2.absdiff(prevFrameImg, curFrameImg)
            difImg = cv2.GaussianBlur(difImg, GAUSIAN_BLUR_SIZE, 0)

        grayDifImg = cv2.cvtColor(difImg, cv2.COLOR_BGR2GRAY)
        # grayDifImg = cv2.dilate(grayDifImg, None, iterations=DILATE_ITER)

        thresh = cv2.inRange(grayDifImg, MOTION_THRESH, 255)
        thresh = cv2.dilate(thresh, None, iterations=DILATE_ITER)

        # Read more about findContours here: https://medium.com/analytics-vidhya/opencv-findcontours-detailed-guide-692ee19eeb18
        contours, hierarch = cv2.findContours(
            thresh, [cv2.RETR_EXTERNAL, cv2.RETR_LIST][0], [cv2.CHAIN_APPROX_NONE, cv2.CHAIN_APPROX_SIMPLE][1])

        drawFrame = currentFrame.copy()
        for cont in contours:
            if cv2.contourArea(cont) > CONTOUR_AREA_THRESH:
                rect = cv2.boundingRect(cont)
                x, y, w, h = rect
                cv2.drawContours(drawFrame, [cont], -1, (0, 255, 0), 3)
                cv2.rectangle(
                    drawFrame, (x, y), (x+w, y+h), (0, 0, 255), 3)

        updateTrajectory(contours)
        for index, trajectory in enumerate(TRAJECTORY_BOUNDS):
            if len(trajectory) > 1:
                for i in range(len(trajectory)-1):
                    random.seed(index)
                    start = rectCenter(trajectory[i])
                    end = rectCenter(trajectory[i+1])
                    cv2.line(
                        drawFrame, start, end, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 3)
        print(len(TRAJECTORY_BOUNDS))

    keyPressed = cv2.waitKey(50)

    if chr(keyPressed & 0xFF) == 'q':
        print('-'*80)
        print('You Pressed [Q] To Quit')
        print('Contour area threshold:\t', CONTOUR_AREA_THRESH)
        print('Motion Threshold:\t', MOTION_THRESH)
        print('Dilation:   \t', DILATE_ITER)
        print('Gausian Blur Size:\t', GAUSIAN_BLUR_SIZE)
        print('Blur Mode:\t', WHEN_BLUR)
        break
    elif chr(keyPressed & 0xFF) == 'p':
        PAUSED = not PAUSED
        print('*'*20, ['\tVideo Feed Has Been Resumed\t',
              '\tVideo Feed Has Been Paused\t'][PAUSED], '*'*20, )
    elif chr(keyPressed & 0xFF) == 'w':
        CONTOUR_AREA_THRESH += 100
        print('Contour area threshold +increased+: \t', CONTOUR_AREA_THRESH)
    elif chr(keyPressed & 0xFF) == 's':
        CONTOUR_AREA_THRESH -= 100
        print('Contour area threshold -decreased-: \t', CONTOUR_AREA_THRESH)
    elif chr(keyPressed & 0xFF) == 'd':
        MOTION_THRESH += 2
        print('Motion Threshold +Increased+: \t', MOTION_THRESH)
    elif chr(keyPressed & 0xFF) == 'a':
        MOTION_THRESH -= 2
        print('Motion Threshold -Decreased-: \t', MOTION_THRESH)
    elif chr(keyPressed & 0xFF) == ']':
        GAUSIAN_BLUR_SIZE = tuple([i+2 for i in GAUSIAN_BLUR_SIZE])
        print('GAUSIAN_BLUR_SIZE +Increased+: \t', GAUSIAN_BLUR_SIZE)
    elif chr(keyPressed & 0xFF) == '[':
        GAUSIAN_BLUR_SIZE = tuple(
            [i-2 if i > 2 else 1 for i in GAUSIAN_BLUR_SIZE])
        print('GAUSIAN_BLUR_SIZE -Decreased-: \t', GAUSIAN_BLUR_SIZE)
    elif chr(keyPressed & 0xFF) == '9':
        DILATE_ITER = DILATE_ITER-1 if DILATE_ITER > 0 else 0
        print('DILATE_ITER *Decreased*: \t', DILATE_ITER)
    elif chr(keyPressed & 0xFF) == '0':
        DILATE_ITER += 1
        print('DILATE_ITER *Increased*: \t', DILATE_ITER)
    elif chr(keyPressed & 0xFF) == 'x':
        WHEN_BLUR = 'ON DIF' if WHEN_BLUR == 'ON FRAME' else 'ON FRAME'
        print('WHEN_BLUR *CHANGED*: \t', WHEN_BLUR)
    elif chr(keyPressed & 0xFF) == 'n':
        playNextFrame = 2
        print('******* NEXT FRAME *******')
    elif keyPressed != -1:
        print(chr(keyPressed & 0xFF), 'Pressed')
        print('Press -W,S- for Contour Threshold')
        print('Press -A,D- for Motion Threshold')
        print('Press -[,]- for Gausain Blur')
        print('Press -9,0- for Dilation')
        print('Press - x - to switch when Blur is Applied')
        print('Press - n - for Next Frame')

    if keyPressed != -1 and chr(keyPressed & 0xFF) != 'n':
        recalculate = True

    if not PAUSED or playNextFrame:
        prevFrameImg = curFrameImg
        playNextFrame -= bool(playNextFrame)

    cv2.resizeWindow('Frame difference Gray', (339, 226))
    cv2.resizeWindow('Thresh', (339, 226))
    cv2.resizeWindow('Contours', (339, 226))
    cv2.resizeWindow('ImgDiff', (339, 226))

    cv2.moveWindow('Frame difference Gray', 0, 0)
    cv2.moveWindow('Thresh', 0, 256)
    cv2.moveWindow('Contours', 0, 512)
    cv2.moveWindow('ImgDiff', 0, 768)

    cv2.imshow('Frame difference Gray', grayDifImg)
    cv2.imshow('Thresh', thresh)
    cv2.imshow('Contours', drawFrame)
    cv2.imshow('ImgDiff', difImg)

cv2.destroyAllWindows()
