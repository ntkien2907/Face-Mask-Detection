# Import all libraries and modules required
import os
import cv2
import numpy as np
from keras.models import load_model

# Load existing model
model = load_model("face-mask-model.h5")

# Label and Bounding Box
results = {0: 'without mask', 1: 'mask'}
GR_dict = {0: (0,0,255), 1: (0,255,0)}
rect_size = 4

# Start video capture
cap = cv2.VideoCapture(0)

# Find out the path to cascade classifier XML file
cascPath = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_default.xml"
haarcascade = cv2.CascadeClassifier(cascPath)

while True:
    rval, im = cap.read()
    im = cv2.flip(im, 1, 1)    
    rerect_size = cv2.resize(im, (im.shape[1]//rect_size, im.shape[0]//rect_size))

    # Convert RGB to grayscale
    gray = cv2.cvtColor(rerect_size, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = haarcascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=4,
        minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    
    for f in faces:
        x, y, w, h = [v*rect_size for v in f] 
        
        face_img = im[y:y+h, x:x+w]
        rerect_sized = cv2.resize(face_img, (150,150))
        normalized = rerect_sized / 255.0
        reshaped = np.reshape(normalized, (1,150,150,3))
        reshaped = np.vstack([reshaped])

        # Predict wearing face-mask or not
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]

        # Draw bounding box
        cv2.rectangle(im, (x,y), (x+w,y+h), GR_dict[label], 2)
        cv2.rectangle(im, (x,y-40), (x+w,y), GR_dict[label], -1)

        # Label "mask" or "without mask"
        cv2.putText(im, results[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    

    cv2.imshow('FACE MASK DETECTION', im)
    key = cv2.waitKey(10)
    
    if key == 27: 
        break

cap.release()
cv2.destroyAllWindows()