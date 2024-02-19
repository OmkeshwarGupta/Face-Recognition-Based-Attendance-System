import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

import pandas as pd

cred = credentials.Certificate("./serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://facedetectionattendances-f2389-default-rtdb.firebaseio.com/"
})

ref= db.reference('Student')

df = pd.read_excel("data.xlsx")

data={}
for index, row in df.iterrows():
    roll_no = row['roll no']
    student_data = {
        "name": row['name'],
        "course": row['course'],
        "batch": row['batch'],
        "total_attendance": row['total_attendance'],
        "last_attendance_time": str(row['last_attendance_time']),
        "attendance": [{"timestamp": str(timestamp)} for timestamp in row['attendance'].split(',')]
    }
    data[roll_no] = student_data

for key, value in data.items():
    ref.child(key).set(value)
    # new_student_ref = ref.push(value)

print(data)

