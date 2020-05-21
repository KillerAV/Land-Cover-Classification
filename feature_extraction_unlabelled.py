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
from numpy import mean, std
from sklearn import svm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt  
from matplotlib import style
import copy

train_path = "/home/vanjani/Desktop/LandCoverClassfication/SampleOfUnlabelledImages/"
train_names = os.listdir(train_path)

with open("/home/vanjani/Desktop/LandCoverClassfication/featurevector_unlabelled.csv","w") as outfile:
    writer=csv.writer(outfile)

number_of_clusters = 6
image_size_x = 1152
image_size_y = 1536
image_size_x = 864
image_size_y = 1152

#Extracts 59 LBP Features
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
        for i in range(256):
            for j in range(256):
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

desc = LocalBinaryPatterns(8,1)

#Extracts 26 (6*4+2) color based features
def extract_color(image, number_of_channels, mask):
    channel_images=[]
    if number_of_channels == 1:
        channel_images.append(image)
    else:
        for i in range(number_of_channels):
            channel_images.append(image[:, :, i])
            
    feature_vector=[]

    #Mean
    for i in range(number_of_channels):
        feature_vector.append(cv2.mean(channel_images[i],mask)[0])
        
    #Standard Deviation
    for i in range(number_of_channels):
        blockx = image.shape[1]
        blocky = image.shape[0]
        tmp1 = np.zeros((blockx, blocky))
        tmp2 = np.zeros((blockx, blocky))
        a, b = cv2.meanStdDev(channel_images[i],tmp1,tmp2,mask)
        feature_vector.append(b[0][0])

    return feature_vector

#Extracts 64 BRIEF Features
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

count=-1;
#Performs Segmentation using K-MEANS
def k_means(image):
    '''
    #FINDING OPTIMAL NUMBER OF CLUSTERS
    cost =[] 
    for i in range(3, 15): 
            KM = KMeans(n_clusters = i, max_iter = 100) 
            KM.fit(resized_image) 

            cost.append(KM.inertia_)      
  
    # plot the cost against K values 
    plt.plot(range(1, 15), cost, color ='g', linewidth ='3') 
    plt.xlabel("Value of K") 
    plt.ylabel("Sqaured Error (Cost)") 
    plt.show() # clear the plot 
    '''
    
    '''
    #EROSION + DILATION
    kernel = np.ones((3,3), np.uint8) 
    img_erosion = cv2.erode(image, kernel, iterations=1) 
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
    image = img_dilation
    '''
    
    resized_image = image.reshape(image.shape[0]*image.shape[1], image.shape[2])/255
    kmeans = KMeans(n_clusters=number_of_clusters,random_state=0).fit(resized_image)
    image_show = kmeans.cluster_centers_[kmeans.labels_]
    cluster_image = image_show.reshape(image.shape[0], image.shape[1], image.shape[2])
    cv2.imwrite(os.path.join("/home/vanjani/Desktop/LandCoverClassfication/",str(count)+".jpg"),(cluster_image*255).astype(np.uint8))
        
    cv2.destroyAllWindows()
    clusters = []
    masks = []
    for i in range(0,number_of_clusters):
        clusters.append(copy.deepcopy(cluster_image[:,:,0]))
        masks.append(copy.deepcopy(image))

    for i in range(0, kmeans.labels_.shape[0]):
        x = int(i/image.shape[1])
        y = int(i%image.shape[1])
        for j in  range(0,number_of_clusters):
            if kmeans.labels_[i]!=j:
                clusters[j][x][y] = 0
                masks[j][x][y] = (0,0,0)
            else:
                clusters[j][x][y] = 255

    
    return clusters, masks
        
print ("[STATUS] Started extracting features")
for name in train_names:
    file=train_path+name
    count+=1
    print(name)
    print("Peforming Image Segmentation")
    original_image = cv2.imread(file)
    #resized_image = cv2.resize(original_image,(640,640))
    resized_image = cv2.resize(original_image,(image_size_y,image_size_x))
    print(resized_image.shape)
    clusters, masks = k_means(resized_image)
    print("Image Segmentation Done")
    
    image = resized_image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsvimage=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    labimage=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
    ycbimage=cv2.cvtColor(image,cv2.COLOR_BGR2YCrCb)
    for i in range(0,number_of_clusters):
        mask = clusters[i].astype(np.uint8)
        masked_image = masks[i]
        image_name = "image"+str(i)+".jpg"
        cv2.imwrite(os.path.join("/home/vanjani/Desktop/LandCoverClassfication/Clusters/"+str(count)+"/",image_name),mask)
        gray_masked_image = cv2.cvtColor(masked_image,cv2.COLOR_BGR2GRAY)
        print("Extracting features of Cluster "+str(i+1))
        temparr=[]
        
        #1. From RGB IMAGE
        features = extract_color(image, 3, mask)
        for i in range(len(features)):
            temparr.append(features[i])

        #2. From HSV IMAGE
        features = extract_color(hsvimage, 3, mask)
        for i in range(len(features)):
            temparr.append(features[i])

        #3. From Gray IMAGE
        features = extract_color(gray, 1, mask)
        for i in range(len(features)):
            temparr.append(features[i])

        #4. From LAB IMAGE
        features = extract_color(labimage, 3, mask)
        for i in range(len(features)):
            temparr.append(features[i])

        #5. From YCrCb IMAGE
        features = extract_color(ycbimage, 3, mask)
        for i in range(len(features)):
            temparr.append(features[i])
        
        #extract LBP features
        hist = desc.describe(gray_masked_image)
        for i in range(len(hist)):
            temparr.append(hist[i])

        #extract BRIEF features
        briefFeatures = extractBriefFeatures(masked_image)
        for i in range(len(briefFeatures)):
            temparr.append(briefFeatures[i])
        
        with open("/home/vanjani/Desktop/LandCoverClassfication/featurevector_unlabelled.csv","a") as outfile:
            writer=csv.writer(outfile)
            writer.writerow(temparr)
