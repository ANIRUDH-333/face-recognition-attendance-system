import os
import time
import cv2
import numpy as np
from PIL import Image
from threading import Thread



# -------------- image labesl ------------------------

def getImagesAndLabels(path):
    # get the path of all the files in the folder
    print("Labels 1")
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # print(imagePaths)
    print("Labels 2")
    # create empth face list
    faces = []
    # create empty ID list
    Ids = []
    # now looping through all the image paths and loading the Ids and the images
    print("Labels 3")
    for imagePath in imagePaths:
        print("Labels 4")
        # loading the image and converting it to gray scale
        pilImage = Image.open(imagePath).convert('L')
        # Now we are converting the PIL image into numpy array
        print("Labels 5")
        imageNp = np.array(pilImage, 'uint8')
        # getting the Id from the image
        print("Labels 6")
        Id = str(os.path.split(imagePath)[-1].split(".")[1])
        print(Id)
        # extract the face from the training image sample
        print("Labels 7")
        faces.append(imageNp)
        print("Labels 8")
        Ids.append((Id))
        print("Labels 9")
        print(faces,Ids)
    return faces, Ids


# ----------- train images function ---------------
def TrainImages():
    print("till here")
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    print("till here 1")
    harcascadePath = "haarcascade_frontalface_default.xml"
    print("till here 2")
    detector = cv2.CascadeClassifier(harcascadePath)
    print("till here 3")
    faces, Id = getImagesAndLabels("TrainingImage")
    print("till here 4")
    Ids = np.array(Id)
    Ids_int = Ids.astype(int)
    Thread(target = recognizer.train(faces, Ids_int)).start()
    print("till here 5")
    # Below line is optional for a visual counter effect
    Thread(target = counter_img("TrainingImage")).start()
    recognizer.save("TrainingImageLabel"+os.sep+"Trainner.yml")
    print("All Images")

# Optional, adds a counter for images trained (You can remove it)
def counter_img(path):
    imgcounter = 1
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    for imagePath in imagePaths:
        print(str(imgcounter) + " Images Trained", end="\r")
        time.sleep(0.008)
        imgcounter += 1

