import  cv2
import  numpy as np
import  face_recognition
import os
from datetime import datetime


# Step 1 - Import images and convert them into rgb
# we create a list that retrive the images from ImageAttendance folder automatically
# and encode it so we can compare it with test images


path = 'C:/Users/Deepak/PycharmProjects/Attendance Project/ImageAttendance'
images = []
classNames = []
mylist = os.listdir(path)
print(mylist)

for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

#  Step 2- Encoding of images
def findEncoding(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return  encodeList

# mark attendance
def markAttendance(name):
    with open('C:/Users/Deepak/PycharmProjects/Attendance Project/Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList =[]
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:

            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
        #print(myDataList)






encodeListKnown = findEncoding(images)

print('Encoding Completed')

# Step 3 - Find mathches between encodings

cap = cv2.VideoCapture(0)

while True:
    # capturing images and convetring them from BRG to RGB
    sucess, img = cap.read()
    imgS = cv2.resize(img,(0,0),None ,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # encoding captured images

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)


    # Finding matches

    for encodeFace , faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # lowest  distance is best match
        #print(faceDis)
        matchIndex = np.argmin(faceDis)
        # now we know which distance is lowest or minimum
        # so now we have to make rectangle boxes around matched faces
        # and display their name

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            #print(name)
            # to draw the rectangle around the faces we need facelocation whis is stored in faceLoc
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4 , y2*4, x1*4
            cv2.rectangle(img , (x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)


        cv2.imshow('Webcame', img)
    if (cv2.waitKey(1) == ord('q')):
            break
cap.release()
cv2.destroyAllWindows()
