#extracts 52+52+26+16+64=210 features of an image
import csv
import cv2
import numpy as np
import os
import glob
import mahotas as mt
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import normalize
import cvutils
from skimage import feature
from scipy.stats import kurtosis, skew
from sklearn import svm

class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        self.numPoints = numPoints
        self.radius = radius
 
    def describe(self, image, eps=1e-7):
        lbp = feature.local_binary_pattern(image, self.numPoints, self.radius, method="default")
        uniform = [0,1,2,3,4,6,7,8,12,14,15,16,24,28,30,31,32,48,56,60,62,63,64,96,112,120,124,126,127,128,129,131,135,143,159,191,192,193,195,199,207,223,224,225,227,231,239,240,241,243,247,248,249,251,252,253,254,255]
        dis = {}
        for i in range(len(uniform)):
            dis[uniform[i]]=i
        hs=[]
        for i in range(59):
            hs.append(0)
        non_uniform_count=0
    
        for i in range(len(lbp)):
            for j in range(len(lbp[i])):
                if lbp[i][j]!=255:
                    if lbp[i][j] in dis:    
                        hs[dis[lbp[i][j]]]+=1
                    else:
                        non_uniform_count+=1
        final_hs=[]
        for i in range(58):
            c=hs[i]
            for j in range(c):
                final_hs.append(i)
        for i in range(non_uniform_count):
            final_hs.append(58)
        hs = final_hs
        hs=np.array(hs)
        (hs,_) = np.histogram(hs, bins=59, weights=np.ones(len(hs)) / len(hs))
        return hs

desc = LocalBinaryPatterns(8,2)
	
train_path = "/home/vanjani/Desktop/LandCoverClassfication/images/"
train_names = os.listdir(train_path)

def extract_color(image, number_of_channels):
    channel_images=[]
    if number_of_channels == 1:
        channel_images.append(image)
    else:
        for i in range(number_of_channels):
            channel_images.append(image[:, :, i])
            
    feature_vector=[]

    #Mean
    for i in range(number_of_channels):
        feature_vector.append(cv2.mean(channel_images[i])[0])
        
    #Standard Deviation
    for i in range(number_of_channels):
        blockx = image.shape[1]
        blocky = image.shape[0]
        tmp1 = np.zeros((blockx, blocky))
        tmp2 = np.zeros((blockx, blocky))
        a, b = cv2.meanStdDev(channel_images[i],tmp1,tmp2)
        feature_vector.append(b[0][0])

    return feature_vector

def extractBriefFeatures(image):
        star = cv2.xfeatures2d.StarDetector_create()
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create(bytes=16)
        vector_size=4
        kps = star.detect(image)
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        kps, des = brief.compute(image, kps)
        needed_size = (vector_size * 16)
        temparr=[]
        for i in range(len(kps)):
                for j in range(0,16):
                        temparr.append(des[i][j])
        while len(temparr) < needed_size:
                for i in range(0,16):
                        temparr.append(0)
        return temparr

input_file=csv.DictReader(open("/home/vanjani/Desktop/LandCoverClassfication/training.csv"))

with open("/home/vanjani/Desktop/LandCoverClassfication/featurevector.csv","w") as outfile:
        writer=csv.writer(outfile)
        
print ("[STATUS] Started extracting features")
count=0;
for row in input_file:
        count+=1;
        if count%100==0:
                print(count)
        file=train_path+row["image"]
        image = cv2.imread(file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsvimage=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        labimage=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
        ycbimage=cv2.cvtColor(image,cv2.COLOR_BGR2YCrCb)
        
        temparr=[]

        #1. From RGB IMAGE
        features = extract_color(image, 3)
        for i in range(len(features)):
                temparr.append(features[i])

        #2. From HSV IMAGE
        features = extract_color(hsvimage, 3)
        for i in range(len(features)):
                temparr.append(features[i])

        #3. From Gray IMAGE
        features = extract_color(gray, 1)
        for i in range(len(features)):
                temparr.append(features[i])

        #4. From LAB IMAGE
        features = extract_color(labimage, 3)
        for i in range(len(features)):
                temparr.append(features[i])

        #5. From YCrCb IMAGE
        features = extract_color(ycbimage, 3)
        for i in range(len(features)):
                temparr.append(features[i])
        
        #extract LBP features
        hist = desc.describe(gray)
        for i in range(len(hist)):
                temparr.append(hist[i])

        #extract BRIEF features
        briefFeatures = extractBriefFeatures(image)
        for i in range(len(briefFeatures)):
                temparr.append(briefFeatures[i])
        
        with open("/home/vanjani/Desktop/LandCoverClassfication/featurevector.csv","a") as outfile:
                writer=csv.writer(outfile)
                writer.writerow(temparr)

