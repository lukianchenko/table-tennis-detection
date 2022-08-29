import cv2
import pandas as pd
from tqdm import tqdm
from pathlib import Path

def yolo_to_corners(coordinates, width=1920, height=1080):
    x_1 = int(width * coordinates[1] - width * coordinates[3] / 2)
    y_1 = int(height * coordinates[2] - height * coordinates[4] / 2)
    x_2 = int(width * coordinates[1] + width * coordinates[3] / 2)
    y_2 = int(height * coordinates[2] + height * coordinates[4] / 2)
    return x_1, y_1, x_2, y_2

def yolo_to_w_h(coordinates, width=1920, height=1080):
    x_1 = int(width * coordinates[1] - width * coordinates[3] / 2)
    y_1 = int(height * coordinates[2] - height * coordinates[4] / 2)
    w = int(width * coordinates[1] + width * coordinates[3] / 2) - x_1
    h = int(height * coordinates[2] + height * coordinates[4] / 2) - y_1
    return x_1, y_1, w, h

def corners_to_w_h(coords):
    return int(coords[0]), int(coords[1]), int(coords[2] - coords[0]), int(coords[3] - coords[1])

def w_h_to_corners(coords):
    return int(coords[0]), int(coords[1]), int(coords[0] + coords[2]), int(coords[1] + coords[3])

VIDEO_FPS = 5
FRAME_SIZE = (1920, 1080)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)

video_path = Path('D:\\Python Projects\\Tennis_Ball_Detection\\results\\source_videos\\train_game_2.avi')
frames_data_path = Path('D:\\Python Projects\\Tennis_Ball_Detection\\results\\train_frames_info.csv')
frames_data = pd.read_csv(frames_data_path)
second_game_df = frames_data[frames_data['game_id'] == 2]

capture = cv2.VideoCapture(str(video_path))
video_cod = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter('D:\\Python Projects\\Tennis_Ball_Detection\\results\\tracked.avi', video_cod, VIDEO_FPS, FRAME_SIZE)

# backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=1000)
# backSub = cv.createBackgroundSubtractorKNN(dist2Threshold=1000, detectShadows=False)

tracker = cv2.TrackerCSRT_create()
backSub = cv2.createBackgroundSubtractorMOG2()

initialized = False

for i, row in tqdm(second_game_df.iterrows(), total=second_game_df.shape[0]):
    ret, frame = capture.read()

    blured_frame = cv2.GaussianBlur(frame, (3, 3), cv2.BORDER_DEFAULT)
    fgMask = backSub.apply(blured_frame)

    if not all(row[['x_1', 'y_1', 'x_2', 'y_2']].isna()):

        if i == 0:
            tracker.init(fgMask, corners_to_w_h(list(row[['x_1', 'y_1', 'x_2', 'y_2']])))
            initialized = True
        else:
            if abs(row['mean'] - second_game_df['mean'].iloc[i-1]) > 0.15:
                tracker.init(fgMask, corners_to_w_h(list(row[['x_1', 'y_1', 'x_2', 'y_2']])))
                initialized = True
                cv2.putText(frame, 'NEW INTERVAL', (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, RED_COLOR, 3)
            else:
                if initialized:
                    found, bbox = tracker.update(fgMask)
                    if found:
                        bbox = w_h_to_corners(bbox)
                        cv2.putText(frame, 'FOUND', (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, GREEN_COLOR, 3)
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), GREEN_COLOR, 3)
                    else:
                        cv2.putText(frame, 'NOT FOUND', (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, RED_COLOR, 3)
                        initialized = False
                else:
                    tracker.init(fgMask, corners_to_w_h(list(row[['x_1', 'y_1', 'x_2', 'y_2']])))
                    initialized = True
    else:
        cv2.putText(frame, 'NO BALL ON FRAME', (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, RED_COLOR, 3)
        initialized = False

    writer.write(frame)

writer.release()
