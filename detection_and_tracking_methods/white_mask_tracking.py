import cv2
from pathlib import Path
import numpy as np

VIDEO_FPS = 5
FRAME_SIZE = (1920, 1080)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)

video_path = Path('D:\\Python Projects\\Tennis_Ball_Detection\\results\\source_videos\\train_game_2.avi')

capture = cv2.VideoCapture(str(video_path))
video_cod = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter('D:\\Python Projects\\Tennis_Ball_Detection\\results\\white_mask_back.avi',
                         video_cod, VIDEO_FPS, FRAME_SIZE, 0)
writer_1 = cv2.VideoWriter('D:\\Python Projects\\Tennis_Ball_Detection\\results\\white_mask.avi',
                           video_cod, VIDEO_FPS, FRAME_SIZE, 0)

backSub = cv2.createBackgroundSubtractorMOG2()

while True:

    ret, frame = capture.read()

    if ret:

        frame = cv2.resize(frame, (500, 300))

        frame_b = cv2.GaussianBlur(frame, (5, 5), cv2.BORDER_DEFAULT)

        hsv = cv2.cvtColor(frame_b, cv2.COLOR_BGR2HSV)
        sensitivity = 170
        lower_white = np.array([0, 0, 255 - sensitivity])
        upper_white = np.array([255, sensitivity, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        fgMask = backSub.apply(frame_b)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        writer.write(mask_white & fgMask)
        writer_1.write(mask_white)

capture.release()
cv2.destroyAllWindows()


