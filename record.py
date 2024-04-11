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

    HAAR_CASCADE = cv.data.haarcascades + "haarcascade_frontalface_default.xml"
        
    #   Open The OpenCV VideoCapture Device to retrieve live images from your webcam (cv.VideoCapture)
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera!")
        exit()
    #variable to detect when to cpature frame 
    frames_since_detection = 0 
    save_frames = True

    while True:
        #capture frame
        ret, frame = cap.read()
         # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        frame_with_rectangle = frame.copy()
        #create cascade
        face_cascade = cv.CascadeClassifier(HAAR_CASCADE)

        #change to grayscale
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # start cascade
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

        #if detected face
        if len(faces) > 0:
            
            for (x,y,w,h) in faces:

                #show rectangle of face
                cv.rectangle(frame_with_rectangle, (x,y), (x+w,y+h), (0,255,0), 2)
                cv.putText(frame_with_rectangle,args,(x,y-10), cv.FONT_HERSHEY_COMPLEX, 0.9 ,(0,255,0), 2)

                if save_frames:

                    #save frame with unique filename
                    filename = f"face_{args}_{uuid.uuid4()}"
                    
                    # save frame with the same filename but with .jpg extension
                    cv.imwrite(os.path.join(target_folder, f"{filename}.jpg"), frame)

                    # write face position to a CSV file with the same filename but with .csv extension
                    with open(os.path.join(target_folder, f"{filename}.csv"), "w", newline="") as csv_file:
                        csv_writer = csv.writer(csv_file, delimiter=",")
                        for x,y,w,h in faces:
                            csv_writer.writerow([x,y,w,h])

                    #change save_frames status
                    save_frames = False

                    frames_since_detection = 0
            #dont save face for 30 frames after face was saved
            else:
                frames_since_detection += 1

                if frames_since_detection >= 30:
                    save_frames = True
        
        #display frame
        cv.imshow('frame', frame_with_rectangle)
    
            #dieses google drive ding

            
        if cv.waitKey(1) == ord('q'):
            break
        
    
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()



    #   Initialize the Haar feature cascade for face recognition from OpenCV (cv.CascadeClassifier)
    #   If the cascade file (haarcascade_frontalface_default.xml) is missing, download it from google drive
    #   Run the cascade on every image to detect possible faces (CascadeClassifier::detectMultiScale)
    #   If there is exactly one face, write the image and the face position to disk in two seperate files (cv.imwrite, csv.writer)
    #   If you have just saved, block saving for 30 consecutive frames to make sure you get good variance of images.
    if args.folder is None:
        print("Please specify folder for data to be recorded into")
        exit()

record("kdkg")