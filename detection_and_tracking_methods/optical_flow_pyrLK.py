import numpy as np
import cv2

VIDEO_FPS = 25
FRAME_SIZE = (1920, 1080)

capture = cv2.VideoCapture("videos/test_game_2.avi")

video_cod = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(
    f'C:/Users/AcerSwift3/Desktop/Abto results/optical_flow_lucas_kanade.mp4', video_cod, VIDEO_FPS, FRAME_SIZE)

ret, old_frame = capture.read()
cv2.imshow('Frame', old_frame)
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

lk_params = dict(winSize=(15, 15),
                 maxLevel=0,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

width = old_frame.shape[1]
height = old_frame.shape[0]
list_of_yolo_data = [0, 0.446875, 0.4601851851851852, 0.00625, 0.011111111111111112]

x1 = int(width * list_of_yolo_data[1] - width * list_of_yolo_data[3] / 2)
y1 = int(height * list_of_yolo_data[2] - height * list_of_yolo_data[4] / 2)
x2 = int(width * list_of_yolo_data[1] + width * list_of_yolo_data[3] / 2)
y2 = int(height * list_of_yolo_data[2] + height * list_of_yolo_data[4] / 2)

# Mouse function
def select_point(event, x, y, flags, params):
    global point, point_selected, old_points
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        point_selected = True
        old_points = np.array([[x, y]], dtype=np.float32)


cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", select_point)

point_selected = True
point = (x1, y1)
old_points = np.array([[x1, y1]], dtype=np.float32)

while capture.isOpened():
    ret, frame = capture.read()
    if frame is None:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(frame_gray, (3, 3), 0)

    if point_selected:
        # calculate optical flow
        new_points, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, blur, old_points, None, **lk_params)
        old_gray = frame_gray.copy()
        old_points = new_points

        x, y = new_points.ravel()
        cv2.circle(frame, (int(x), int(y)), 10, (0, 0, 255), 4)
    cv2.imshow('Frame', frame)
    writer.write(frame)

    if cv2.waitKey(1) == 27:
        break

writer.release()
capture.release()
cv2.destroyAllWindows()