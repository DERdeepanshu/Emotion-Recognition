import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from keras.preprocessing import image
import os
from pickle import load


def video_recog():
	model = model_from_json(open("fer2.json", "r").read())
	model.load_weights('fer2.h5')
	faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
	font = cv2.FONT_HERSHEY_SIMPLEX
	label_list = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
	id = 0
	cam = cv2.VideoCapture(0)
	cam.set(3, 640)
	cam.set(4, 480)
	minW = 0.1*cam.get(3)
	minH = 0.1*cam.get(4)
	while cam.isOpened():
	    ret, frame = cam.read()
	    frame = cv2.flip(frame, 1)
	    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	    faces = faceCascade.detectMultiScale(gray, 1.2, 5, minSize=(int(minW), int(minH)))
	    for (x, y, w, h) in faces:
	        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
	        roi_gray = gray[y:y+h, x:x+w]
	        roi_gray = cv2.resize(roi_gray, (48, 48))
	        img_pixels = image.img_to_array(roi_gray)
	        img_pixels = np.expand_dims(img_pixels, axis=0)
	        img_pixels /= 255
	        prediction = model.predict(img_pixels)
	       # if confidence < 100:
	        #    id = label_list[id]
	         #   confidence = "  {0}%".format(round(100-confidence))
	        #else:
	         #   id = "unknown"
	          #  confidence = "  {0}%".format(round(100-confidence))
	        max_index = np.argmax(prediction[0])
	        cv2.putText(frame, label_list[max_index], (x+5, y-5), font, 1, (255, 255, 255), 2)
	        #cv2.putText(frame, str(confidence*100), (x+5, y+h+5), font, 1, (255, 255, 0), 1)
	    cv2.imshow('video', frame)
	    if cv2.waitKey(1) & 0xFF == ord('q'):
	        break
	cam.release()

def image_recog(img_path):
	model = model_from_json(open("fer.json", "r").read())
	model.load_weights('fer.h5')
	scaler = load(open('scaler.pkl', 'rb'))
	faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
	font = cv2.FONT_HERSHEY_SIMPLEX
	label_list = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
	minW = 0.1*640
	minH = 0.1*480
	print(img_path)
	img = cv2.imread(img_path)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(gray, 1.2, 5, minSize=(int(minW), int(minH)))
	for (x, y, w, h) in faces:
	    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
	    roi_gray = gray[y:y+h, x:x+w]
	    roi_gray = cv2.resize(roi_gray, (48, 48))
	    img_pixels = image.img_to_array(roi_gray)
	    img_pixels = np.expand_dims(img_pixels, axis=0)
	    img_pixels /= 255
	    prediction = model.predict(img_pixels)
	    max_index = np.argmax(prediction[0])
	    cv2.putText(img, label_list[max_index], (x+5, y-5), font, 1, (255, 255, 255), 2)
	    #cv2.putText(img, str(confidence*100), (x+5, y+h+5), font, 1, (255, 255, 0), 1)
	cv2.imshow('Image', img)
	cv2.waitKey(0)
	cam.release()