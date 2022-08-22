import cv2 as cv

backSub = cv.createBackgroundSubtractorMOG2()
# backSub = cv.createBackgroundSubtractorKNN()
    
capture = cv.VideoCapture("train_game_1.avi")
if not capture.isOpened():
    print('Unable to open video')
    exit(0)
    
while True:
    ret, frame = capture.read()
    if frame is None:
        break
    
    blured_frame = cv.GaussianBlur(frame, (3,3), cv.BORDER_DEFAULT)
    
    fgMask = backSub.apply(blured_frame)
    
    cv.imshow('Frame', blured_frame)
    cv.imshow('FG Mask', fgMask)
    
    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break