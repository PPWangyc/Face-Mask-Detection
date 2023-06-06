# USAGE
# python detect_mask_video.py

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream, FileVideoStream, FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			if face.any():
				face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
				face = cv2.resize(face, (224, 224))
				face = img_to_array(face)
				face = preprocess_input(face)

				# add the face and bounding boxes to their respective
				# lists
				faces.append(face)
				locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-v", "--video", type=str, default="/home/yanchen/Data/breathe/s156/s156_session 6_12-21-21_part02.mp4", help="path to optional video file")

# define a function: write the box and label to a file
def write_to_file(output_path, frame_num, label, box):
	with open(output_path + "/label.txt", "a") as f:
		f.write(str(frame_num) + " " + label + " " + str(box[0]) + " " + str(box[1]) + " " + str(box[2]) + " " + str(box[3]) + "\n")

def write_value_to_file(output_path, total_frame, total_time, fps):
	with open(output_path + "/label_value.txt", "a") as f:
		f.write("total_frame:{}, total_time:{}, fps:{}\n".format(total_frame, total_time, fps))

# create a directory where the video is stored
def create_dir(file_name):
	output_path = file_name.split("/")[0:-1]
	output_path = "/".join(output_path)
	output_path = os.path.join(output_path,file_name.split("/")[-1])
	if not os.path.exists(output_path):
		os.makedirs(output_path)
	return output_path

args = vars(ap.parse_args())
file_name = args["video"].split(".")[0]
# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video file thread...")

# fvs = FileVideoStream(args["video"]).start()

output_path = create_dir(file_name)

time.sleep(1.0)

count = 0

cap = cv2.VideoCapture(args["video"])
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("FPS:", fps)
print("Total Frames:", total_frame)
total_time=total_frame/fps
print("Total Time:", total_time)
write_value_to_file(output_path, total_frame=total_frame, total_time=total_time, fps=fps)
# loop over the frames from the video stream
while cap.isOpened():
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	ret, frame = cap.read()
	if frame is None:
		break
	frame = imutils.resize(frame, width=400)

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
	if len(locs) == 0:
		print("no face detected")
		count += 5
		cap.set(cv2.CAP_PROP_POS_MSEC, count * 1000)
		continue
	# loop over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# determine the class label and color we'll use to draw
		# the bounding box and text
		# if mask: label = 1, if no mask: label = 0
		label = 1 if mask > withoutMask else "0"
		# color = (0, 255, 0) if label == 1 else (0, 0, 255)
			
		# include the probability in the label
		label = "{}: {:.4f}".format(label, max(mask, withoutMask))

		# display the label and bounding box rectangle on the output
		# frame
		# cv2.putText(frame, label, (startX, startY - 10),
		# 	cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		# cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		# crop the face according to the box
		frame = frame[startY:endY, startX:endX]
	cframe = cap.get(cv2.CAP_PROP_POS_FRAMES)
	print(type(frame))
	try:
		cv2.imwrite(os.path.join(output_path,'{:d}.jpg'.format(count)), frame)
		write_to_file(output_path, count, label, box)
	except:
		count += 5
	count += 5
	cap.set(cv2.CAP_PROP_POS_MSEC, count * 1000)

cap.release()
