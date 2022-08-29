import cv2
import numpy as np

# Constants.
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.25
DIFF_THRESHOLD = 40

# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
THICKNESS = 1

# Colors
BLACK  = (0, 0, 0)
BLUE   = (255, 178, 50)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
ORANGE = (0, 165, 255)

VIDEO_FPS = 25
FRAME_SIZE = (1920, 1080)

def draw_label(input_image, label, left, top):
	"""Draw text onto image at location."""

	# Get text size.
	text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
	dim, baseline = text_size[0], text_size[1]
	# Use text size to create a BLACK rectangle. 
	cv2.rectangle(input_image, (left, top), (left + dim[0], top + dim[1] + baseline), BLACK, cv2.FILLED);
	# Display text inside the rectangle.
	cv2.putText(input_image, label, (left, top + dim[1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)

def pre_process(input_image, net):
	# Create a 4D blob from a frame.
	blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), [0,0,0], 1, crop=False)

	# Sets the input to the network.
	net.setInput(blob)

	# Runs the forward pass to get output of the output layers.
	output_layers = net.getUnconnectedOutLayersNames()
	outputs = net.forward(output_layers)
	# print(outputs[0].shape)

	return outputs

def post_process(input_image, outputs):
	# Lists to hold respective values while unwrapping.
	class_ids = []
	confidences = []
	boxes = []

	# Rows.
	rows = outputs[0].shape[1]

	image_height, image_width = input_image.shape[:2]

	# Resizing factor.
	x_factor = image_width / INPUT_WIDTH
	y_factor =  image_height / INPUT_HEIGHT

	# Iterate through detections.
	for r in range(rows):
		row = outputs[0][0][r]
		confidence = row[4]

		# Discard bad detections and continue.
		if confidence >= CONFIDENCE_THRESHOLD:
			classes_scores = row[5:]

			# Get the index of max class score.
			class_id = np.argmax(classes_scores)

			#  Continue if the class score is above threshold.
			if (classes_scores[class_id] > SCORE_THRESHOLD):
				confidences.append(confidence)
				class_ids.append(class_id)

				cx, cy, w, h = row[0], row[1], row[2], row[3]

				left = int((cx - w/2) * x_factor)
				top = int((cy - h/2) * y_factor)
				width = int(w * x_factor)
				height = int(h * y_factor)
			  
				box = np.array([left, top, width, height])
				boxes.append(box)

	return boxes, confidences, class_ids

def draw_boxes(input_image, boxes, confidences, class_ids, old_box, color, new_interval):

	# Perform non maximum suppression to eliminate redundant overlapping boxes with
	# lower confidences.
	if len(confidences) > 0:
		indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
	elif len(boxes) > 0:
		indices = [0]
	else:
		indices = []

	box = np.empty(0, dtype=np.int32)
	
	for i in indices:
		box = boxes[i]
		if len(old_box) != 0 and not new_interval:
			x1, y1 = box[:2]
			x2, y2 = old_box[:2]
			length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
			if length > DIFF_THRESHOLD:
				box = []
				break
		left = box[0]
		top = box[1]
		width = box[2]
		height = box[3]
		cv2.rectangle(input_image, (left, top), (left + width, top + height), color, 3*THICKNESS)
		if len(confidences) > 0:
			label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])
			draw_label(input_image, label, left + width, top + height)

	return input_image, box

def bacground_subtraction_method(old_box, img, fgMask, new_interval):
	
	# Crop image, zoom in to the ball
	if len(old_box) > 0:
		x1, y1, w, h = old_box
		
		x1 = x1 - 250
		x2 = x1 + w + 250
		y1 = y1 - 250
		y2 = y1 + h + 250

		if x1 < 0 or y1 < 0 or x2 > fgMask.shape[1] or y2 > fgMask.shape[0]:
			fgMask = cv2.copyMakeBorder(fgMask, - min(0, y1), max(y2 - fgMask.shape[0], 0),
								-min(0, x1), max(x2 - fgMask.shape[1], 0),cv2.BORDER_REPLICATE)
			y2 += -min(0, y1)
			y1 += -min(0, y1)
			x2 += -min(0, x1)
			x1 += -min(0, x1)	
		mask = np.zeros(fgMask.shape[:2], dtype=np.uint8)
		cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
		fgMask = cv2.bitwise_and(fgMask, fgMask, mask=mask)
	
	contours, hierarchy = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		
	boxes2 = []
	for cnt in contours:
		approx = cv2.approxPolyDP(cnt, 0.03 * cv2.arcLength(cnt, True), True)
		area = cv2.contourArea(cnt)

		if len(approx) == 8 and area > 130 and area < 250:
			box2 = cv2.boundingRect(cnt)
			boxes2.append(np.array(box2))	

	img, box = draw_boxes(img, boxes2, [], [], old_box, ORANGE, new_interval)

	return img, box.copy(), fgMask

if __name__ == '__main__':
	# Load class names.
	classesFile = "classes.names"
	classes = None
	with open(classesFile, 'rt') as f:
		classes = f.read().rstrip('\n').split('\n')

	cap = cv2.VideoCapture("videos/test_game_2.avi")
	ret, frame = cap.read()
	length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 
	# Initialize background subtractor
	backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=100)
	tracker = cv2.TrackerCSRT.create()	
	
	# Give the weight files to the model and load the network using them.
	modelWeights = "weights.onnx"
	net = cv2.dnn.readNet(modelWeights)
	old_box = np.empty(0, dtype=np.int32)
	old_img = np.empty_like(frame, dtype=np.uint8)
	old_fgMask = np.empty_like(frame, dtype=np.uint8)
	old_interval_diff = 0

	video_cod = cv2.VideoWriter_fourcc(*'mp4v')
	writer = cv2.VideoWriter(f'C:/Users/AcerSwift3/Desktop/Abto results/final.mp4', video_cod, VIDEO_FPS, FRAME_SIZE)

	while cap.isOpened():
		
		ret, frame = cap.read()
		# if frame is read correctly ret is True
		if not ret:
			break
		
		# Process image.
		detections = pre_process(frame, net)
		boxes, confidences, class_ids  = post_process(frame.copy(), detections)

		img = frame.copy()
  
		# Work with background subtitution
		blurred_img = cv2.GaussianBlur(img, (15,15), 3)
		fgMask = backSub.apply(blurred_img)

		# Process new interval
		new_interval_diff = np.mean(cv2.absdiff(img, old_img))
		new_interval = (new_interval_diff - old_interval_diff) > 1
		old_interval_diff = new_interval_diff
  
		img, box = draw_boxes(img, boxes, confidences, class_ids, old_box, GREEN, new_interval)
		if len(box) != 0:
			old_box = box.copy()
		elif len(old_box) > 0: 
			boxes2 = []
			tracker.init(old_fgMask, tuple(old_box))
			found, box2 = tracker.update(fgMask)
			boxes2.append(np.array(box2))
			if found:
				img, box = draw_boxes(img, boxes2, [], [], old_box, RED, new_interval)
				old_box = box.copy()
			else:
				img, old_box, fgMask= bacground_subtraction_method(old_box, img, fgMask, new_interval)
		else:
			img, old_box, fgMask = bacground_subtraction_method(old_box, img, fgMask, new_interval)

		cv2.imshow('Output', img)
		writer.write(img)
		# cv2.imshow('fgMask', fgMask)
  
		old_img = img.copy()
		old_fgMask = fgMask.copy()
		
		if cv2.waitKey(1) == 27:
			break
		
	# When everything done, release the capture
	cap.release()
	writer.release()
	cv2.destroyAllWindows()
