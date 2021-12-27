import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load existing model
model = load_model("face-mask-model.h5")

# Label and Bounding Box
results = {0: 'incorrectly mask', 1: 'with mask', 2: 'without mask'}
GR_dict = {0: (255,0,0), 1: (0,255,0), 2: (0,0,255)}
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
    faces = haarcascade.detectMultiScale(gray, 
                                         scaleFactor=1.05, 
                                         minNeighbors=5, 
                                         minSize=(40,40), 
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    
    for face in faces:
        # The coordinates of the face
        x, y, w, h = [v*rect_size for v in face]
        face_im = im[y:y+h, x:x+w]
        
        # Resize frame to 224x224 and normalize
        resized = cv2.resize(im, (224,224))
        normalized = resized / 255.0
        
        # Expand dimensions from (224,224,3) to (1,224,224,3)
        reshaped = np.expand_dims(normalized, axis=0)
        
        # Predict wearing face-mask or not
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]
        print(result)

        # Draw bounding box
        cv2.rectangle(im, (x,y), (x+w,y+h), GR_dict[label], 2)
        cv2.rectangle(im, (x,y-40), (x+w,y), GR_dict[label], -1)

        # Put label name on bounding box
        cv2.putText(im, results[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    
    cv2.imshow('Face-mask Detection', im)
    key = cv2.waitKey(10)
    
    if key == 27: 
        break

cap.release()
cv2.destroyAllWindows()
