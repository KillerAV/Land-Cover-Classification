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
import copy


"""
accuracy=(tp+tn)/(tp+tn+fp+fn)
sensitivity=tp/(tp+fn)=recall
specificity=tn/(tn+fp)
TP= C1 classified to C1
TN= Non C1 classified to Non C1
FP= Non C1 classified to C1
FN= C1 classified to non C1
precision=tp/(tp+fp)
"""
#To print the analysis and results
def print_analysis(expected,prediction):
    acc=[]
    sens=[]
    speci=[]
    precision=[]
    total=0
    for j in range(3):
        tp=0
        tn=0
        fp=0
        fn=0
        for key,value in expected.items():
            expected_output=value
            actual_output=prediction[key]
            if expected_output==int(j):
                if actual_output==int(j):
                    tp+=1
                else:
                    fn+=1
            else:
                if actual_output==int(j):
                    fp+=1
                else:
                    tn+=1

        acc.append((tp+tn)/(tn+tp+fn+fp))
        sens.append(tp/(tp+fn))
        speci.append(tn/(tn+fp))
        precision.append(tp/(tp+fp))
        total+=tp
        
    for i in range(3):
        print ("class "+str(i))
        print ("Accuracy: " + str(acc[i]))
        print ("Sensitivity: "+ str(sens[i]))
        print ("Specificity: "+ str(speci[i]))
        print ("Precision: "+str(precision[i]))
        print ("Recall: "+ str(sens[i]))

        
expected_output={}
image_name="image_4"

#Vegetation Images
path = "/home/vanjani/Desktop/LandCoverClassfication/accuracy_of_segmentation/classified_images/"+image_name+"/Vegetation/"
names = os.listdir(path)

for file in names:
    expected_output[file]=0

#Residential Images
path = "/home/vanjani/Desktop/LandCoverClassfication/accuracy_of_segmentation/classified_images/"+image_name+"/Residential/"
names = os.listdir(path)

for file in names:
    expected_output[file]=1

#Sea/Lake Images
path = "/home/vanjani/Desktop/LandCoverClassfication/accuracy_of_segmentation/classified_images/"+image_name+"/Sea/"
names = os.listdir(path)

for file in names:
    expected_output[file]=2

actual_output={}
path="/home/vanjani/Desktop/LandCoverClassfication/accuracy_of_segmentation/input_images/"
original_image = cv2.imread(path+image_name+".jpg")
path="/home/vanjani/Desktop/LandCoverClassfication/accuracy_of_segmentation/images_after_classification/"
image = cv2.imread(path+image_name+".jpg")
original_image = cv2.resize(original_image,(image.shape[1],image.shape[0]))

blockSizeX=54
blockSizeY=72
count=0
segmented_image_result = copy.deepcopy(image)
inix = 0
for i in range(0,len(image),blockSizeX):
    iniy = 0
    for j in range(0,len(image[i]),blockSizeY):
        small_image=image[i:i+blockSizeX,j:j+blockSizeY]
        image_name = "image_"+str(count)+".jpg"
        count+=1
        color=(0,0,0)
        original_color=(0,0,0)
        vegetation_count, residential_count, sea_count = (0,0,0)
        for x in range(len(small_image)):
            for y in range(len(small_image[0])):
                if small_image[x][y][0]<=10 and small_image[x][y][1]>=245 and small_image[x][y][2]<=245:
                    vegetation_count+=1
                elif small_image[x][y][0]>=117 and small_image[x][y][0]<=137 and small_image[x][y][1]>=117 and small_image[x][y][1]<=137 and small_image[x][y][2]>=117 and small_image[x][y][2]<=137:
                    residential_count+=1
                elif small_image[x][y][0]>=250 and small_image[x][y][1]<=5 and small_image[x][y][2]<=5:
                    sea_count+=1
        
        if vegetation_count>=sea_count and vegetation_count>=residential_count:
            color=(0,255,0)
            actual_output[image_name]=0
        elif residential_count>=vegetation_count and residential_count>=sea_count:
            color=(127,127,127)
            actual_output[image_name]=1
        else:
            color=(255,0,0)
            actual_output[image_name]=2

        if expected_output[image_name]==0:
            original_color=(0,255,0)
        elif expected_output[image_name]==1:
            original_color=(127,127,127)
        else:
            original_color=(255,0,0)

        for x in range(inix,inix+blockSizeX):
            for y in range(iniy,iniy+blockSizeY):
                segmented_image_result[x][y]=color
                original_image[x][y]=original_color
                
        iniy+=blockSizeY
    inix+=blockSizeX

print_analysis(expected_output, actual_output)


                    
    
        



        
