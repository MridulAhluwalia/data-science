#!/usr/bin/env python
import cv2
import time
import pickle
import imutils
import numpy as np

from flask import Flask
from flask import Response
from flask import render_template
from datetime import datetime
from imutils.video import VideoStream
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# -------------------- CONST --------------------

PROTO_TXT = 'detector/deploy.prototxt'
CAFFE_MODEL = 'detector/res10_300x300_ssd_iter_140000.caffemodel'
CNN_MODEL = 'models/liveness.model'
LE_OBJ = 'models/le.pickle'
CONFIDENCE = .7

# -------------------- INIT --------------------

app = Flask(__name__)

net = cv2.dnn.readNetFromCaffe(PROTO_TXT, CAFFE_MODEL)
model = load_model(CNN_MODEL)
le = pickle.loads(open(LE_OBJ, "rb").read())

vs = VideoStream(src=0).start()
time.sleep(2)

# -------------------- FUNC --------------------

def generate():
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=600)

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        	(300, 300), (104.0, 177.0, 123.0))

        net.setInput(blob)
        detections = net.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > CONFIDENCE:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)

                face = frame[startY:endY, startX:endX]
                face = cv2.resize(face, (32, 32))
                face = face.astype("float") / 255.0
                face = img_to_array(face)
                face = np.expand_dims(face, axis=0)

                preds = model.predict(face)[0]
                j = np.argmax(preds)
                label = le.classes_[j]

                confidence_text = '%s: %.2f' % (label, preds[j])

                cv2.putText(frame, confidence_text, (startX, startY - 10),
    				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                	(0, 0, 255), 2)

        timestamp = datetime.now()
        cv2.putText(frame,
            timestamp.strftime("%A %d %B %Y %I:%M:%S%p"),
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        videoFrame = frame.copy()
        (flag, encodedImage) = cv2.imencode(".jpg", videoFrame)
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
    		bytearray(encodedImage) + b'\r\n')

# -------------------- ROUTE --------------------

@app.route("/")
def home():
	return render_template("home.html")

@app.route("/video_feed")
def video_feed():
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

# -------------------- START --------------------

if __name__ == '__main__':
    app.run(debug=True, threaded=True, use_reloader=False)

vs.stop()
