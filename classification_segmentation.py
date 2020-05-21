import csv
import cv2
import numpy as np
import os
import random
from sklearn import svm
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.ndimage import imread

x_train=[]
y_train=[]
x_test=[]
y_test=[]
class_label=["Vegetation", "Residential", "Sea"]
numberOfClusters=6
image_size_x = 1152
image_size_y = 1536
image_size_x = 864
image_size_y = 1152

#USING LINEAR SVM CLASSIFIER
def linear_svm_classifier(x_train,y_train,x_test):
    global y_test
    svm_model_linear = SVC(gamma='scale', probability=True).fit(x_train, y_train) 
    y_test = svm_model_linear.predict(x_test) 
    
#USING RBF SVM CLASSIFIER
def rbf_svm_classifier(x_train,y_train,x_test):
    global y_test
    svm_model_linear = SVC(kernel = 'rbf', gamma='scale', C = 10, probability=True).fit(x_train, y_train) 
    y_test = svm_model_linear.predict(x_test)
    
#USING RANDOM FOREST CLASSIFIER
def random_forest_classifier(x_train,y_train,x_test):
    global y_test
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(x_train, y_train)
    y_test = clf.predict(x_test)

#SHOWING THE CLASSIFIED CLUSTERS
def show_clusters(result, image_number):
    cluster_path = "/home/vanjani/Desktop/LandCoverClassfication/Clusters/"+str(image_number)+"/"
    cluster_names = os.listdir(cluster_path) 
    mapping = {}
    for i in range(numberOfClusters):
        mapping["image"+str(i)+".jpg"]=i

    output_image = []
    for i in range(image_size_x):
        temp=[]
        for j in range(image_size_y):
            temp.append((0,0,0))
        output_image.append(temp)
        
    for name in cluster_names:
        file = cluster_path+name
        index = mapping[name]
        color = (255,255,255)
        if result[index]==0:
            color = (0,255,0)
        elif result[index]==1:
            color = (127,127,127)
        else:
            color = (255,0,0)
        image = imread(file)
        print(file)
        for i in range(image_size_x):
            for j in range(image_size_y):
                if image[i][j]>=250:
                    output_image[i][j] = color
        
    output_image = np.array(output_image)
    output_image = output_image.astype(np.uint8)
    output_path = "/home/vanjani/Desktop/LandCoverClassfication/segmentation_results/"
    cv2.imwrite(os.path.join(output_path,"output_"+str(image_number)+".jpg"), output_image)
    
def classifier_result(x_train, y_train, x_test):
    global y_test

    print("RANDOM-FOREST")
    random_forest_classifier(x_train,y_train,x_test)
    print(y_test)

    image_number=4
    for i in range(0,len(y_test),numberOfClusters):
        result=[]
        for j in range(i,i+numberOfClusters):
            result.append(y_test[j])
        show_clusters(result,image_number)
        image_number+=1

with open('/home/vanjani/Desktop/LandCoverClassfication/featurevector.csv', newline='') as myFile:
    reader = csv.reader(myFile)
    for row in reader:
        tempvec=[]
        for i in range(len(row)):
            tempvec.append(row[i])
        x_train.append(tempvec)

with open('/home/vanjani/Desktop/LandCoverClassfication/featurevector_unlabelled.csv', newline='') as myFile:
    reader = csv.reader(myFile)
    for row in reader:
        tempvec=[]
        for i in range(len(row)):
            tempvec.append(row[i])
        x_test.append(tempvec)
        
with open('/home/vanjani/Desktop/LandCoverClassfication/training.csv',newline='') as file:
    reader = csv.reader(file)
    flag=1
    for row in reader:
        if flag==1:
            flag=0
            continue
        y_train.append(row[1])

for i in range(len(y_train)):
    y_train[i]=int(y_train[i])
    
for i in range(len(x_train)):
    for j in range(len(x_train[i])):
        x_train[i][j]=float(x_train[i][j])
        
for i in range(len(x_test)):
    for j in range(len(x_test[i])):
        x_test[i][j]=float(x_test[i][j])

x_train=np.array(x_train)
y_train=np.array(y_train)
x_test=np.array(x_test)

'''
Color: 0-25
LBP: 26-84
BRIEF: 85-148
'''
'''
#COLOR
mapping=[]
for i in range(0,26):
    mapping.append(i)
    
arr=mapping
x1=x_train[:,arr]
x2=x_test[:,arr]

print("COLOR")
classifier_result(x1,y_train,x2)
'''
'''
#LBP
mapping=[]
for i in range(26,85):
    mapping.append(i)
    
arr=mapping
x1=x_train[:,arr]
x2=x_test[:,arr]

print("LBP")
classifier_result(x1,y_train,x2)

#BRIEF
mapping=[]
for i in range(85,149):
    mapping.append(i)
    
arr=mapping
x1=x_train[:,arr]
x2=x_test[:,arr]

print("BRIEF")
classifier_result(x1,y_train,x2)
'''
#COLOR+LBP
mapping=[]
for i in range(0,85):
    mapping.append(i)
    
arr=mapping
x1=x_train[:,arr]
x2=x_test[:,arr]

print("COLOR+LBP")
classifier_result(x1,y_train,x2)
'''
#COLOR+BRIEF
mapping=[]
for i in range(0,26):
    mapping.append(i)
for i in range(85,149):
    mapping.append(i)
    
arr=mapping
x1=x_train[:,arr]
x2=x_test[:,arr]

print("COLOR+BRIEF")
classifier_result(x1,y_train,x2)
'''
'''
#COLOR+LBP+BRIEF
mapping=[]
for i in range(0,149):
    mapping.append(i)
    
arr=mapping
x1=x_train[:,arr]
x2=x_test[:,arr]

print("COLOR+LBP+BRIEF")
classifier_result(x1,y_train,x2)
'''
