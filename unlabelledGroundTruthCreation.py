import cv2
import os

# load the training dataset
train_path = "/home/vanjani/Desktop/LandCoverClassfication/SampleOfUnlabelledImages/"
train_names = os.listdir(train_path)
destination_path = "/home/vanjani/Desktop/LandCoverClassfication/ClassifiedLandCoverImages/"

for name in train_names:
    file=train_path+name
    print(name)
    image = cv2.imread(file)
    print(image.shape)
    blockSizeX=384
    blockSizeY=512
    for i in range(0,len(image),blockSizeX):
        for j in range(0,len(image[i]),blockSizeY):
            small_image=image[i:i+blockSizeX,j:j+blockSizeY]
            cv2.imwrite(os.path.join(destination_path,"image_"+str(count)+".jpg"), small_image)
            count+=1
