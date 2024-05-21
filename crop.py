import cv2 as cv
from common import ROOT_FOLDER, TRAIN_FOLDER, VAL_FOLDER
import os
import csv
import random
import numpy as np
from PIL import Image

root_directory = os.getcwd() #this is the root folder of the program

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

    # clean folders from all files if there are any
    # for root, dirs, files in os.walk(ROOT_FOLDER):
    #     for name in files:
    #         os.remove(os.path.join(root, name))

    #create object folder
    if not os.path.exists(ROOT_FOLDER):
        os.mkdir(ROOT_FOLDER)

    #create the train, val and "person-name"-folders
    train_folder_path = os.path.join(root_directory, TRAIN_FOLDER)
    val_folder_path = os.path.join(root_directory, VAL_FOLDER)
    for folder in [train_folder_path, val_folder_path]:
        if not os.path.exists(folder):
            os.mkdir(folder)
                
    # iterate over all object folders
    for root, dirs, files in os.walk(ROOT_FOLDER):
        for dir_name in dirs:
            object_folder = os.path.join(root, dir_name) #take the current folder as the "object_folder" for this iteration
            print(object_folder)
            # iterate over the full images
            for file in os.listdir(object_folder):
                file = os.path.join(object_folder, file)
                print(f"I'm in file {file} now")

                #calculate borders if it's a csv-file
                if file.endswith(".csv"):
                    crop_width, crop_height, x1, y1, x2, y2 = border_calculation(file, args.border)
                    
                #crop the actual image with given border-size if it's an image-file
                elif file.endswith(('.jpg', '.jpeg', '.png')):
                    frame = cv.imread(file)

                    # apply the given border with BORDER_REFLECT
                    frame_with_border = cv.copyMakeBorder(frame, crop_height, crop_height, crop_width, crop_width, cv.BORDER_REFLECT)
                    
                    #create a folder of the person in the train/val folders and set output directory to train or val
                    output_folder = train_folder_path if random.uniform(0.0, 1.0) < float(args.split) else val_folder_path
                    person_folder = os.path.join(output_folder, dir_name)
                    if not os.path.exists(person_folder):
                        os.mkdir(person_folder)

                    # cache-save frame_with_border to train/val
                    cv.imwrite(os.path.join(person_folder, file), frame_with_border)

                    print(x1, y1, x2, y2)
                    image_array = np.array(frame_with_border)
                    # Crop the image by slicing the array
                    cropped_image_array = image_array[y1:y2, x1:x2]
                    # Convert the cropped array back to an image
                    frame_cropped = Image.fromarray(cropped_image_array)
                    frame_cropped_array = np.array(frame_cropped)

                    #open, crop and save image
                    #frame_to_crop = Image.open(frame_with_border) 
                    #frame_cropped = frame_to_crop.crop((x1, y1, x2, y2))
                    #frame_cropped.save(person_folder)
                    #cv.imwrite(person_folder, frame_cropped)

                    #cropped_image = frame_with_border[y1:y2, x1:x2] 
                    print(f"cropping of {file} completed.")

                    # show the cropped image
                    cv.imshow('Cropped Image', frame_cropped_array)
                    cv.waitKey(0)
                    cv.destroyAllWindows()

                    #THIS IS COMMENTED SO I CAN TEST THE CODE BETTER:
                    #delete uncropped image from objects-folder
                    #os.remove(file)
                    #print(f"removed {file} from object folder.")


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