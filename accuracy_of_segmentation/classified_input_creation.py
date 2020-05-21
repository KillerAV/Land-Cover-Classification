import cv2
import os

# load the training dataset
train_path = "/home/vanjani/Desktop/LandCoverClassfication/accuracy_of_segmentation/input_images/"
train_names = os.listdir(train_path)
destination_path = "/home/vanjani/Desktop/LandCoverClassfication/accuracy_of_segmentation/classified_images/"

image_count=3
for name in train_names:
    file=train_path+name
    print(name)
    if name!="image_4.jpg":
        continue
    image_count+=1
    image = cv2.imread(file)
    image = cv2.resize(image,(1152, 864))
    print(image.shape)
    blockSizeX=54
    blockSizeY=72
    count=0
    for i in range(0,len(image),blockSizeX):
        for j in range(0,len(image[i]),blockSizeY):
            small_image=image[i:i+blockSizeX,j:j+blockSizeY]
            cv2.imwrite(os.path.join(destination_path+"image_"+str(image_count)+"/","image_"+str(count)+".jpg"), small_image)
            count+=1
