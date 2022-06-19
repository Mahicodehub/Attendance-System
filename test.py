import cv2
import face_recognition
import os
import numpy as np
import pandas as pd
import pickle



data = pickle.loads(open('face_enc1', "rb").read()) #to load the training encodings of all faces in the folder


df = pd.read_csv('/home/mahi/Desktop/kl/Major project/train.csv')#to load the actual data of all students for reference
idn = df['ID Number'].tolist()
nam = df['Name'].tolist()

def namePrint(name):  #to print name corresponds to id number
    if name in idn:
        row=nam[idn.index(name)]
        return row





img = cv2.imread('/home/mahi/Desktop/test3.jpeg') #group image for testing


imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #to change color from Rgb to Bgr
faces_in_frame = face_recognition.face_locations(imgS) #to recognize the all faces in the image
#faces_in_frame = faces
print("Number of faces detected: ",(len(faces_in_frame)))
#print(len(faces_in_frame))
encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)#to find the encodings of all faces in particular image
l = []
for encode_face, faceloc in zip(encoded_faces,faces_in_frame):  
    matches = face_recognition.compare_faces(data["encodings"], encode_face) #to compare face encodings from test image to train images
   # print(matches)
    faceDist = face_recognition.face_distance(data["encodings"], encode_face) #to find the face distance for test image to train images
    #print(faceDist)
    matchIndex = np.argmin(faceDist)     #to find the minimum distance for all compare images
    #print(faceDist[matchIndex])

    if matches[matchIndex] and faceDist[matchIndex]<0.5:  #to print particular name on the face in image
        classNames = data["names"]
        ids = classNames[matchIndex].upper()
        #print(ids)
        l.append(ids)
        name = namePrint(ids)
        y1,x2,y2,x1 = faceloc
        #y1, x2,y2,x1 = y1*2,x2*2,y2*2,x1*2
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.rectangle(img, (x1,y2),(x2,y2), (0,255,0), cv2.FILLED)
        cv2.putText(img,name, (x1,y2), cv2.FONT_HERSHEY_COMPLEX,0.65,(100,200,255),1)
    else:                               #if face is not in train images it declared as unknown
        y1, x2, y2, x1 = faceloc
        y1, x2, y2, x1 = y1, x2, y2, x1
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img,"unknown", (x1,y2), cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,0),1)
print(l)
m=[]
for i in l:     #to record the attendence data of all faces in the image
    if i in idn:
        row=df.iloc[idn.index(i)]
        m.append(row)   
print(m)
df1 = pd.DataFrame(m)



#df1.to_csv("Att.csv")
#df = pd.DataFrame(l)
df1.to_csv('Attendence.csv')  #to print the attendence sheet for presentees
cv2.imshow("img",img)
cv2.waitKey(0)

