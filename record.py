import numpy as np
import cv2 as cv
import os
import gdown
import uuid
import csv
from common import ROOT_FOLDER, GOOGLE_DRIVE_LINK


# This is the data recording pipeline
def record(args):

    # Exit if folder is None
    if args.folder is None:
        print("Please specify folder for data to be recorded into")
        exit()

    # Create folder for recorded person
    target_folder = os.path.join(ROOT_FOLDER, args.folder)
    os.makedirs(target_folder, exist_ok=True)

    HAAR_CASCADE = cv.data.haarcascades + "haarcascade_frontalface_default.xml"

    # Google drive download
    if not os.path.isfile(HAAR_CASCADE):
        print("File not found. Downloading from google drive...")
        url = GOOGLE_DRIVE_LINK
        output = "haarcascade_frontalface_default.xml"
        gdown.download(url, output, fuzzy=True)
        HAAR_CASCADE = output

    # Open webcam
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera!")
        exit()

    # Variable to detect when to save frame
    frames_since_detection = 0
    save_frames = True

    while True:
        # Capture frame
        ret, frame = cap.read()
        # If frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame_with_rectangle = frame.copy()
        # Create cascade
        face_cascade = cv.CascadeClassifier(HAAR_CASCADE)

        # Change to grayscale
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Start cascade
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

        # If detected face
        if len(faces) > 0:
            for (x, y, w, h) in faces:

                # Show rectangle of face
                cv.rectangle(frame_with_rectangle, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv.putText(frame_with_rectangle, args.folder, (x, y-10), cv.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)

        if len(faces) == 1 and save_frames:

            # Save frame with unique filename
            filename = f"face_{args.folder}_{uuid.uuid4()}"

            # Save frame with the same filename but with .jpg extension
            cv.imwrite(os.path.join(target_folder, f"{filename}.jpg"), frame)

            # Write face position to a CSV file with the same filename but with .csv extension
            with open(os.path.join(target_folder, f"{filename}.csv"), "w", newline="") as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=",")
                for x, y, w, h in faces:
                    csv_writer.writerow([x, y, w, h])

            # Change save_frames status and reset counter
            save_frames = False
            frames_since_detection = 0

        # Dont save face for 30 frames after face was saved
        else:
            frames_since_detection += 1

            if frames_since_detection >= 30:
                save_frames = True

        # Display frame
        cv.imshow('frame', frame_with_rectangle)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
