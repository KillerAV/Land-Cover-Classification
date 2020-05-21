import csv
import cv2
import numpy as np
import os
import imutils
import skimage
from scipy import ndarray
from skimage import transform
from skimage import util
import random
from matplotlib import pyplot

with open("/home/vanjani/Desktop/LandCoverClassfication/training.csv","w") as outfile:
    writer=csv.writer(outfile)

actual_image=[]
image_name=[]
output=[]

#Vegetation Images
path = "/home/vanjani/Desktop/LandCoverClassfication/ClassifiedLandCoverImages/Vegetation/"
names = os.listdir(path)

for file in names:
    image=cv2.imread(path+file)
    actual_image.append(image)
    image_name.append(file)
    output.append(0)

#Residential Images
path = "/home/vanjani/Desktop/LandCoverClassfication/ClassifiedLandCoverImages/Residential/"
names = os.listdir(path)

for file in names:
    image=cv2.imread(path+file)
    actual_image.append(image)
    image_name.append(file)
    output.append(1)

#Sea/Lake Images
path = "/home/vanjani/Desktop/LandCoverClassfication/ClassifiedLandCoverImages/Sea/"
names = os.listdir(path)

for file in names:
    image=cv2.imread(path+file)
    actual_image.append(image)
    image_name.append(file)
    output.append(2)

with open("/home/vanjani/Desktop/LandCoverClassfication/training.csv","a") as outfile:
    writer=csv.writer(outfile)
    writer.writerow(["image","output"])
    path="/home/vanjani/Desktop/LandCoverClassfication/images/"
    for i in range(len(image_name)):
        writer.writerow([image_name[i],output[i]])
        cv2.imwrite(os.path.join(path,image_name[i]), actual_image[i])

