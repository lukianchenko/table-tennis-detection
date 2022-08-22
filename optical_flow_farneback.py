import numpy as np
import cv2 as cv

cap = cv.VideoCapture("train_game_1.avi")
ret, frame1 = cap.read()
if not ret:
    print('No frame1 grabbed!')
    exit(0)
    
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

while True:
    ret, frame2 = cap.read()
    
    cv.imshow("Blured", frame2)
    if not ret:
        print('No frames grabbed!')
        break

    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 4, 5, 1, 7, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    cv.imshow('frame2', bgr)
    
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    
    prvs = next

cv.destroyAllWindows()