import cv2
import numpy as np

VIDEO_FPS = 25
FRAME_SIZE = (1920, 1080)

# backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=1000)
backSub = cv2.createBackgroundSubtractorKNN(
    dist2Threshold=500, detectShadows=False)

capture = cv2.VideoCapture("videos/test_game_2.avi")
video_cod = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(
    f'C:/Users/AcerSwift3/Desktop/Abto results/background_substraction_mask.mp4', video_cod, VIDEO_FPS, FRAME_SIZE, False)

while capture.isOpened():
    ret, frame = capture.read()
    if frame is None:
        break

    blur = cv2.GaussianBlur(frame, (11, 11), 0)
    fgMask = backSub.apply(blur)

    contours, hierarchy = cv2.findContours(
        fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_list = []
    for cnt in contours:
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        area = cv2.contourArea(cnt)

        # len(approx) >=8 and cv2.isContourConvex(cnt)
        if area > 100 and area < 400 :
            (x, y, w, h) = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)
            contour_list.append(cnt)

    # cv2.drawContours(frame, contour_list, -1, (0, 0, 255), 5)
    cv2.imshow('Frame', frame)
    cv2.imshow('FG Mask', fgMask)

    # writer.write(fgMask)

    if cv2.waitKey(1) == 27:
        break

capture.release()
writer.release()
cv2.destroyAllWindows()
