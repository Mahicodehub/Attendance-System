from imutils import paths
import face_recognition
import pickle
import cv2
import os

#to find the face encodings of all images in the folder for training the model

path="/home/mahi/Desktop/kl/Major project/faces"  #path to images folder
images = []
classNames = []
mylist = os.listdir(path)
for cl in mylist:           #Read the all images from folder one by one
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])   #to extract the names of images.
#print(classNames)
#print(images)
def findEncodings(images):          #To find the encodings of each fcae in folder
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded_face = face_recognition.face_encodings(img)[0]
        encodeList.append(encoded_face)
    return encodeList
encoded_face_train = findEncodings(images) /#to store the encodings of faces

data = {"encodings": encoded_face_train, "names": classNames}
print(data["encodings"])
#use pickle to save data into a file for later use
f = open("face_enc1", "wb")
f.write(pickle.dumps(data))
f.close()
print("Face encodings completed!!!!")
