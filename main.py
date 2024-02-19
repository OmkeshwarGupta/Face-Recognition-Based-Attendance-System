
            

import cv2
import os
import pickle
import face_recognition
import numpy as np
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
from datetime import datetime


cred = credentials.Certificate("./serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://facedetectionattendances-f2389-default-rtdb.firebaseio.com/",
    'storageBucket': "facedetectionattendances-f2389.appspot.com"
})
bucket = storage.bucket()

# url="http://10.41.185.67:4747/video"
cap=cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

imgBackground = cv2.imread("./Resources/background.png")

# Importing the mode images into a list
folderModePath='Resources/Modes/'
modePath=os.listdir(folderModePath)

imgModeList=[]

for path in modePath:
    imgModeList.append(cv2.imread(os.path.join(folderModePath,path)))

# Load the encoding part
print("Loading Encoding File...")
file=open('EncodeFile.p','rb')
encodeListKnownWithId=pickle.load(file)
file.close()
encodeListKnown,studentId=encodeListKnownWithId
# print(studentId)
print("Encoding File Loaded")

modeType=0
counter=0
id=-1
imgStudent = []

while True :
  success, img = cap.read()
  imgS=cv2.resize(img,(0,0),None,0.25,0.25)
  imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
  
  facesCurFrame=face_recognition.face_locations(imgS)
  encodeCurFrame=face_recognition.face_encodings(imgS,facesCurFrame)
  

  imgBackground[162:162+480,55:55+640]=img
  imgBackground[44:44+633,808:808+414]=imgModeList[modeType]
  
  if facesCurFrame:
    for encodeFace, faceLoc in zip(encodeCurFrame,facesCurFrame):
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis=face_recognition.face_distance(encodeListKnown,encodeFace)
        matchIndex=np.argmin(faceDis)
        print("faceDis", faceDis)
        print("MatchIndex",matchIndex)

        if matches[matchIndex] and faceDis[matchIndex] < 0.4: 
            # Process the recognized face
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            # Draw rectangle around the recognized face
            imgBackground = cv2.rectangle(imgBackground, (55 + x1, 162 + y1), (55 + x2, 162 + y2), (0, 255, 0), 2)
            imgBackground = cv2.rectangle(imgBackground, (55 + x1, 162 + y2 - 25), (55 + x2, 162 + y2), (0, 255, 0),
                                        cv2.FILLED)
            cv2.putText(imgBackground, studentId[matchIndex], (55 + x1 + 6, 162 + y2 - 6), cv2.FONT_HERSHEY_COMPLEX,
                        0.5, (0, 0, 0), 1)
            id = studentId[matchIndex]
            # print(studentId[matchIndex])

            if (counter == 0):
                counter = 1
                modeType = 1
            
        else:
            # Handle unrecognized faces
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            # Draw rectangle around the unrecognized face
            imgBackground = cv2.rectangle(imgBackground, (55 + x1, 162 + y1), (55 + x2, 162 + y2), (255, 0, 0), 2)
            imgBackground = cv2.rectangle(imgBackground, (55 + x1, 162 + y2 - 25), (55 + x2, 162 + y2), (255, 0, 0),
                                        cv2.FILLED)
            cv2.putText(imgBackground, 'Unknown', (55 + x1 + 6, 162 + y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        (0, 0, 0), 1)
            
  if(counter!=0):
        
        if(counter==1):
            
            studentInfo=db.reference(f'Student/{id}').get()
            blob = bucket.get_blob(f'Images/{id}.jpg')
            array = np.frombuffer(blob.download_as_string(), np.uint8)
            imgStudent = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)
            datetimeObject = datetime.strptime(studentInfo['last_attendance_time'],
                                                    "%Y-%m-%d %H:%M:%S")
            secondsElapsed = (datetime.now() - datetimeObject).total_seconds()
            if secondsElapsed > 30:
                ref=db.reference(f'Student/{id}')
                attendance_count = len(studentInfo.get('attendance', []))  # Calculate the attendance count
            
                ref.update({
                    'total_attendance': studentInfo['total_attendance']+1,
                    'last_attendance_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    f'attendance/{attendance_count}/timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
           
                })
            else:
                modeType=3
                counter=0
                imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

        if modeType!=3:
                    
            if 10<counter<=20:
                modeType=2
                imgBackground[44:44+633,808:808+414]=imgModeList[modeType]


            if(counter<=10):


                cv2.putText(imgBackground,str(studentInfo['total_attendance']),(861,125),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1)
                cv2.putText(imgBackground,str(studentInfo['course']),(1006,550),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
                cv2.putText(imgBackground,str(id),(1006,493),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
                (w, h), _ = cv2.getTextSize(studentInfo['name'], cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
                offset = (414 - w) // 2
                cv2.putText(imgBackground, str(studentInfo['name']), (808 + offset, 445),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 1)
                

                imgBackground[175:175 + 216, 909:909 + 216]=imgStudent

            counter+=1
            if counter>=20:
                counter=0
                modeType=0
                imgStudent = []
                id=-1
                imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]   
  else:
        modeType = 0
        counter = 0                  
              
#   cv2.imshow("Webcam", img)
  cv2.imshow("Face Detection", imgBackground)
  
  cv2.waitKey(1)
