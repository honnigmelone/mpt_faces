import numpy as np
import cv2 as cv
import os
import gdown
import uuid
import csv
from common import ROOT_FOLDER
#from cascade import create_cascade

# Quellen
#  - How to open the webcam: https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
#  - How to run the detector: https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html
#  - How to download files from google drive: https://github.com/wkentaro/gdown
#  - How to save an image with OpenCV: https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html
#  - How to read/write CSV files: https://docs.python.org/3/library/csv.html
#  - How to create new folders: https://www.geeksforgeeks.org/python-os-mkdir-method/

# This is the data recording pipeline
def record(args):
    # TODO: Implement the recording stage of your pipeline
    #   Create missing folders before you store data in them (os.mkdir)
    
    # Create missing folders before you store data in them
    target_folder = os.path.join(ROOT_FOLDER, args)
    os.makedirs(target_folder, exist_ok=True)

    counter = 0
    face_match = False
        
    #   Open The OpenCV VideoCapture Device to retrieve live images from your webcam (cv.VideoCapture)
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera!")
        exit()

    while True:
        #capture frame
        ret, frame = cap.read()
         # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        # Display the resulting frame
        cv.imshow('frame', frame)

        #detect and save face
        if counter == 30:

            pass
        if cv.waitKey(1) == ord('q'):
            break
        
    
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()



    #   Initialize the Haar feature cascade for face recognition from OpenCV (cv.CascadeClassifier)
    face_cascade = cv.CascadeClassifier(HAAR_CASCADE)
    #   If the cascade file (haarcascade_frontalface_default.xml) is missing, download it from google drive
    #   Run the cascade on every image to detect possible faces (CascadeClassifier::detectMultiScale)
    #   If there is exactly one face, write the image and the face position to disk in two seperate files (cv.imwrite, csv.writer)
    #   If you have just saved, block saving for 30 consecutive frames to make sure you get good variance of images.
    if args.folder is None:
        print("Please specify folder for data to be recorded into")
        exit()

record("kdkg")