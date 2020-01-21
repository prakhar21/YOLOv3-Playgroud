"""
Date: 20th Jan'2020
@author: Prakhar

Summary: Object Detection inference using Pre-trained YOLOv3 weights.
"""

import numpy as np
import cv2
import os

BASE_DIR = '.'
LABELS_PATH = os.path.abspath(os.path.join(BASE_DIR, 'labels.txt'))
LABELS = open(LABELS_PATH).read().strip().split("\n")
COLORS = np.random.uniform(0, 255, size=(len(LABELS), 3))

WEIGHTS = os.path.abspath(os.path.join(BASE_DIR, 'yolov3.weights'))
CONFIG = os.path.abspath(os.path.join(BASE_DIR, 'yolov3.cfg'))

CONFIDENCE = 0.5

#Loading Pre-trained Model
DNN_TARGET = cv2.dnn.DNN_TARGET_CPU
net = cv2.dnn.readNetFromDarknet(CONFIG, WEIGHTS)
net.setPreferableTarget(DNN_TARGET);

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# output_layers = [output_layers[0]]

def detect(image):
	"""
	Summary:
		Reads the image and runs pre-trained YOLO for object detection.
	Params:
		image: Image for testing.
	Returns:
		Saves annotated image to disk under "detected" folder.
	"""
	#read image using cv2 and resize
	IMG_NAME = image.split('/')[-1]
	img = cv2.imread(image)
	img = cv2.resize(img, None, fx=0.4, fy=0.4)

	#accessing attributes for the image
	# will be usefull later while making box around objects
	height, width, _ = img.shape

	#make blog for allowed size for yolo input
	blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416)) #320, 416, 609

	# do a forward pass 
	#layer_out holds the predictions
	net.setInput(blob)
	layer_out = net.forward(output_layers)
	
	boxes = []
	confidences = []
	classes = []

	for output in layer_out:
		for detection in output: #detection == len(labels)
			scores = detection[5:] #first 4 are reserved for x,y,w,h
			class_found = np.argmax(scores)
			confidence = scores[class_found]

			if confidence > CONFIDENCE:
				center_x = int(detection[0] * width)
				center_y = int(detection[1] * height)
				w = int(detection[2] * width)
				h = int(detection[3] * height)

				x = int(center_x - (w / 2))
				y = int(center_y - (h / 2))
				boxes.append([x,y,w,h])
				confidences.append(float(confidence))
				classes.append(class_found)

	idxs = cv2.dnn.NMSBoxes(bboxes=boxes, scores=confidences, score_threshold=CONFIDENCE, nms_threshold=0.4)

	if len(idxs) > 0:
		for i in idxs[0]:
			
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
	 
			color = [int(c) for c in COLORS[classes[i]]]
			cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classes[i]], confidences[i])
			cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	cv2.imwrite(os.path.abspath(os.path.join(os.path.join(BASE_DIR, 'detected'), IMG_NAME)), img)
	return

detect('./raw/dog.jpg')
