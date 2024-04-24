import cv2 as cv
from common import ROOT_FOLDER, TRAIN_FOLDER, VAL_FOLDER
import os
import csv
import random

# Quellen
#  - How to iterate over all files/folders in one directory: https://www.tutorialspoint.com/python/os_walk.htm
#  - How to add border to an image: https://www.geeksforgeeks.org/python-opencv-cv2-copymakeborder-method/

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
    for root, dirs, files in os.walk(ROOT_FOLDER):
        for name in files:
            os.remove(os.path.join(root, name))

    #create folders if they don't exist yet
    if not os.path.exists(ROOT_FOLDER):
        os.mkdir(ROOT_FOLDER)
    train_folder_path = os.path.join(ROOT_FOLDER, TRAIN_FOLDER)
    val_folder_path = os.path.join(ROOT_FOLDER, VAL_FOLDER)
    for folder in [train_folder_path, val_folder_path]:
        if not os.path.exists(folder):
            os.mkdir(folder)
                
    # iterate over all object folders
    for root, dirs, files in os.walk(ROOT_FOLDER):
        for dir_name in dirs:
            object_folder = os.path.join(root, dir_name) #here smth is not quite well solved, need to work on this..

            # iterate over the full images
            for root, dirs, files in os.walk(object_folder):
                for file in files:
                    print(f"I'm in file {file} now")
                    #calculate borders if it's a csv-file
                    if file.endswith(".csv"):
                        crop_width, crop_height = border_calculation(os.path.join(object_folder, file), args.border)
                    
                    #crop the actual image with given border-size
                    elif file.endswith(('.jpg', '.jpeg', '.png')):
                        frame = cv.imread(os.path.join(object_folder, file))

                        # border
                        frame = cv.copyMakeBorder(frame, crop_width, crop_width, crop_height, crop_height, cv.BORDER_REFLECT)

                        # set the output directory for the new cropped image-file
                        output_folder = train_folder_path if random.uniform(0.0, 1.0) < float(args.split) else val_folder_path

                        # save cropped image to the output directory
                        cv.imwrite(os.path.join(output_folder, file), frame)

                        print(f"cropping of {file} completed.")

                        #delete uncropped image from objects-folder
                        os.remove(os.path.join(object_folder, file))
                        print(f"removed {file} from object folder.")

def border_calculation(image_path, border=0.2):
    #open csv and save the coordinates to variable "coords"
    with open(image_path, "r", encoding="utf-8-sig") as file:
        csv_reader = csv.reader(file, delimiter=",")
        coords = list(csv_reader)
    
    # get coordinates from csv
    # only works with csv looking like this:
    # top_left_x,top_left_y; top_right_x,top_right_y; bottom_right_x,bottom_right_y; bottom_left_x,bottom_left_y
    top_left_x, top_left_y = map(int, coords[0])
    bottom_right_x, bottom_right_y = map(int, coords[2])
    
    # calculate width and height of the image
    original_width = bottom_right_x - top_left_x
    original_height = bottom_right_y - top_left_y
    
    # calculate borders based on the set border-%
    crop_width = int(original_width * border)
    crop_height = int(original_height * border)

    print(crop_width)
    print(crop_height)
    
    return crop_width, crop_height


#parser for test
#parser = argparse.ArgumentParser(description='Crop images')
#parser.add_argument('--border', type=float, default=0.2, help='Border value for cropping')
#parser.add_argument('--split', type=float, help='Split value for deciding whether to save in train or val folder')
#args = parser.parse_args()
#border_calculation("/Users/mjy/croptest/objects/marie/marieistdas.jpeg")
#python crop.py --border 0.2 --split 0.8