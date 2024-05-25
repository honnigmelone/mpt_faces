import cv2 as cv
from common import ROOT_FOLDER, TRAIN_FOLDER, VAL_FOLDER
import os
import csv
import random
import numpy as np
from PIL import Image


# This is the cropping of images
def crop(args):
    # TODO: Crop the full-frame images into individual crops
    #   Create the TRAIN_FOLDER and VAL_FOLDER is they are missing (os.mkdir)
    #   Clean the folders from all previous files if there are any (os.walk)
    #   Iterate over all object folders and for each such folder over all full-frame images 
    #   Read the image (cv.imread) and the respective file with annotations you have saved earlier (e.g. CSV)
    #   Attach the right amount of border to your image (cv.copyMakeBorder)
    #   Crop the face with border added and save it to either the TRAIN_FOLDER or VAL_FOLDER
    #   You can use 
    #
    #       random.uniform(0.0, 1.0) < float(args.split) 
    #
    #   to decide how to split them.

    if args.border is None:
        print("Cropping mode requires a border value to be set")
        exit()

    args.border = float(args.border)
    if args.border < 0 or args.border > 1:
        print("Border must be between 0 and 1")
        exit()


    #create the train, val and "person-name"-folders
    train_folder_path = os.path.join(os.getcwd(), TRAIN_FOLDER)
    val_folder_path = os.path.join(os.getcwd(), VAL_FOLDER)
    
    for folder in [train_folder_path, val_folder_path]:
        if not os.path.exists(folder):
            os.mkdir(folder)

        for root,dirs, files in os.walk(folder):
            for file in files:
                filepath = os.path.join(root, file)
                if os.path.isfile(filepath):
                    os.remove(filepath)

    # iterate over all object folders
    for root, dirs, files in os.walk(ROOT_FOLDER):
        for dir_name in dirs:
            object_folder = os.path.join(root, dir_name)
            for file_name in os.listdir(object_folder):
                filepath = os.path.join(object_folder, file_name)
                
                #calculate borders if it's a csv-file
                if filepath.endswith(".csv"):
                    crop_width, crop_height, x1, y1, x2, y2 = border_calculation(filepath, args.border)
                    
                #crop the actual image with given border-size if it's an image-file
                elif filepath.endswith(('.jpg', '.jpeg', '.png')):
                    frame = cv.imread(filepath)
        

                    # apply the given border with BORDER_REFLECT
                    frame_with_border = cv.copyMakeBorder(frame, crop_height, crop_height, crop_width, crop_width, cv.BORDER_REFLECT)
                    #create a folder of the person in the train/val folders and set output directory to train or val
                    output_folder = train_folder_path if random.uniform(0.0, 1.0) > float(args.split) else val_folder_path
                    person_folder = os.path.join(output_folder, dir_name)
                    if not os.path.exists(person_folder):
                        os.mkdir(person_folder) 

                    frame_with_border_path = os.path.join(person_folder, file_name)
                    cv.imwrite(frame_with_border_path, frame_with_border)
        
                    # Crop and save the image
                    cropped_image = frame_with_border[y1:y2, x1:x2]
                    output_path = os.path.join(person_folder, file_name)
                    cv.imwrite(output_path, cropped_image)
                    print(f"Cropped and saved: {output_path}")


# This is the calculation of the border by the given input
def border_calculation(csv_path, border):
    #open csv and save the coordinates to variable "coords"
    with open(csv_path, "r", encoding="utf-8-sig") as file:
        csv_reader = csv.reader(file, delimiter=",")
        coords = next(csv_reader)
    
    x = float(coords[0])
    y = float(coords[1])
    w = float(coords[2])
    h = float(coords[3])
    border = float(border)

    print(f"OLD COORDS/width ARE: {x, y, w, h}")

    # calculate borders based on the set border-%
    new_crop_width = int(w * border)
    new_crop_height = int(h * border)

    print(f"NEW CROP WIDTH/HEIGHT IS: {new_crop_width, new_crop_height}")

    # get coords for the actual cropping
    x1 = int(x + new_crop_width)
    y1 = int(y + new_crop_height)
    x2 = int(x + w + new_crop_width)
    y2 = int(y + h + new_crop_height)

    print(f"NEW COORDS ARE: {x1, y1, x2, y2}")

    return new_crop_width, new_crop_height, x1, y1, x2, y2