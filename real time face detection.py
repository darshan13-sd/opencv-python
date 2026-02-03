
import cv2
import numpy as np

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Open webcam
videocapture = cv2.VideoCapture(0)

scale_factor = 1.2

while True:
    ret, pic = videocapture.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scale_factor, 5)

    # Draw rectangle & text
    for (x, y, w, h) in faces:
        cv2.rectangle(pic, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(
            pic,
            'Darshan',
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

    print("Number of faces found:", len(faces))

    cv2.imshow('Face Detection', pic)

    # Press ESC to exit
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

videocapture.release()
cv2.destroyAllWindows()

