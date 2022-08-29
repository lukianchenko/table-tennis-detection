import cv2
import numpy as np

VIDEO_FPS = 25
FRAME_SIZE = (1920, 1080)


def draw_flaw(img, flow, step=20):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def draw_hsv(flow):
    (h, w) = flow.shape[:2]
    (fx, fy) = (flow[:, :, 0], flow[:, :, 1])
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx + fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(v*10, 0xFF)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow("hsv", bgr)
    return bgr


if __name__ == "__main__":
    capture = cv2.VideoCapture("../tennis ball detection/videos/test_game_2.avi")
    ret, img = capture.read()

    video_cod = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(
        f'C:/Users/AcerSwift3/Desktop/Abto results/optical_flow_farneback.mp4', video_cod, VIDEO_FPS, FRAME_SIZE)

    prevgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    show_hsv = True
    show_glitch = False
    cur_glitch = img.copy()

    while capture.isOpened():
        (ret, img) = capture.read()
        vis = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prevgray, gray, None, 0.5, 10, 20, 3, 5, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        prevgray = gray
        cv2.imshow('flow', draw_flaw(gray, flow))

        if show_hsv:
            hsv = draw_hsv(flow)
            gray_hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
            kernel = np.array([[-1, -1, -1],
                               [-1, 5, -1],
                               [-1, -1, -1]])
            gray_hsv = cv2.filter2D(gray_hsv, -1, kernel)

            contours, hierarchy = cv2.findContours(
                gray_hsv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

            contour_list = []
            for cnt in contours:
                epsilon = 0.01 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                area = cv2.contourArea(cnt)

                if len(approx) > 8 and area > 20 and area < 250:
                    (x, y, w, h) = cv2.boundingRect(cnt)
                    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 5)
                    contour_list.append(cnt)

            cv2.imshow("Image", vis)
            # writer.write(hsv)

        if cv2.waitKey(1) == 27:
            break

capture.release()
writer.release()
cv2.destroyAllWindows()