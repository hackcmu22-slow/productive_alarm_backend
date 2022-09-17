# NORMAL VIDEO STREAMING

# OpenCV and Flask modules
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from flask import Flask, Response

# Flask App
app = Flask(__name__)
# setup camera and resolution
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# GENERATING NORMAL VIDEO FEED FRAMES
def gen_video_frames():  
    while True:
        success, frame = cam.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

# Routing normal video feed
@app.route("/")
def video_feed():
    return Response(gen_video_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# STREAMING WITH YOLO (COMMENT OUT EVERYTHING BELOW THIS TO RUN WITHOUT YOLO)

# Import Yolo Modules
import argparse
import os
import sys
from pathlib import Path
import torch

# Loading the model
model = torch.hub.load('.', 'yolov5s', source='local', skip_validation=True)
model.classes = [0, 79]

# Defining booleans
contains_person_and_toothbrush = False

# This function is to get the yolo streaming.
def gather_yolo(contains_person):
    while True:
        time.sleep(0.1)
        _, img = cam.read()
        try:
            results = model(img)
            results.render()
            _, frame = cv2.imencode('.jpg', results.imgs[0])
            predictions = results.pandas().xyxy[0] #Printing the predictions
            print(predictions["class"][0])
            try:
                # Person Class: 0, Toothbrush == 79
                if (predictions["class"][0] == 0 and predictions["class"][1] == 79):
                    contains_person_and_toothbrush = True 
                if contains_person_and_toothbrush == True: 
                    # Replace print("Wow!") to post a request to the success page.
                    print("WOW!")
            except:
                print("List index out of range")
            # print(type(predictions))
            # print(predictions)
            # Returning if a human and a toothbrush are detected.
        except Exception as e:
            print(e)
            _, frame = cv2.imencode('.jpg', img)
        yield (b'--frame\r\nContent-Type: imageviewer/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n')

@app.route("/yolofeed")
def yolo_mjpeg():
    return Response(gather_yolo(contains_person), mimetype='multipart/x-mixed-replace; boundary=frame')
