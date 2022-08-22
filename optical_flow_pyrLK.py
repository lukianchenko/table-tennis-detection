import numpy as np
import cv2 as cv


cap = cv.VideoCapture("train_game_1.avi")
# Take first frame and find corners in it
ret, old_frame = cap.read()
cv.imshow('Frame', old_frame)
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 0,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Mouse function
def select_point(event, x, y, flags, params):
    global point, point_selected, old_points
    if event == cv.EVENT_LBUTTONDOWN:
        point = (x, y)
        point_selected = True
        old_points = np.array([[x, y]], dtype=np.float32)

cv.namedWindow("Frame")
cv.setMouseCallback("Frame", select_point)

point_selected = False
point = ()
old_points = np.array([])


while True:
    ret, frame = cap.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    if point_selected:
        # calculate optical flow
        new_points, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, old_points, None, **lk_params)
        old_gray = frame_gray.copy()
        old_points = new_points
        
        x, y = new_points.ravel()
        cv.circle(frame, (int(x), int(y)), 5, (0, 0, 255))
        cv.imshow('Frame', frame)
        
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

    
cv.destroyAllWindows()