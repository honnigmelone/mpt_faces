import cv2 as cv
import torch
import os
from network import Net
from cascade import create_cascade
from transforms import ValidationTransform
from PIL import Image

# NOTE: This will be the live execution of your pipeline

HAAR_CASCADE = cv.data.haarcascades + "haarcascade_frontalface_default.xml"

def live(args):
    # TODO: 
    #   Load the model checkpoint from a previous training session (check code in train.py)
    checkpoint = torch.load('model.pt')

    model = Net(len(checkpoint['classes']))

    model.load_state_dict(checkpoint['model'])

    model.eval()

    #   Also, create a video capture device to retrieve live footage from the webcam.
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

    #   Attach border to the whole video frame for later cropping.
    border_size = int(min(frame.shape[:2]) * float(args.border))
    frame_with_border = cv.copyMakeBorder(frame, border_size, border_size, border_size, border_size, cv.BORDER_REFLECT, value=[0, 0, 0])


    #   Initialize the face recognition cascade again (reuse code if possible)
    face_cascade = cv.CascadeClassifier(HAAR_CASCADE)  

    gray_frame = cv.cvtColor(frame_with_border, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    frame_with_rectangle = frame_with_border.copy()
    if len(faces) > 0:    
            for (x,y,w,h) in faces:
                #show rectangle of face
                cv.rectangle(frame_with_rectangle, (x,y), (x+w,y+h), (0,255,0), 2)
                cv.putText(frame_with_rectangle,args,(x,y-10), cv.FONT_HERSHEY_COMPLEX, 0.9 ,(0,255,0), 2)

                # retrieve picture + crop




    #   Run the cascade on each image, crop all faces with border.



    #   Run each cropped face through the network to get a class prediction.



    #   Retrieve the predicted persons name from the checkpoint and display it in the image



    if args.border is None:
        print("Live mode requires a border value to be set")
        exit()

    args.border = float(args.border)
    if args.border < 0 or args.border > 1:
        print("Border must be between 0 and 1")
        exit()