import cv2
import numpy as np

backSub = cv2.createBackgroundSubtractorMOG2()
# backSub = cv.createBackgroundSubtractorKNN(dist2Threshold=1000, detectShadows=False)

capture = cv2.VideoCapture("../tennis ball detection/videos/train_game_2.avi")
if not capture.isOpened():
    print('Unable to open video')
    exit(0)
    
# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 100
params.maxThreshold = 255

# Filter by Area.
params.filterByArea = True
params.minArea = 0
params.maxArea = 500

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.9

# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.2

# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.01

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(frame, (55, 55), 0)
    fgMask = backSub.apply(blur)
    cv2.imshow('Original FG Mask', fgMask)
    
    # Detect blobs.
    keypoints = detector.detect(fgMask)
    # Draw detected blobs as red circles.
    im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    cv2.imshow('Frame', im_with_keypoints)

    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break

capture.release()
cv2.destroyAllWindows()
