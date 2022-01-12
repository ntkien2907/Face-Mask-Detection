import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load existing model
model = load_model('face-mask-model.h5')

# Label and Bounding Box
results = {0: 'incorrectly mask', 1: 'with mask', 2: 'without mask'}
GR_dict = {0: (255,0,0), 1: (0,255,0), 2: (0,0,255)}

# Start video capture
cap = cv2.VideoCapture(0)

# Find out the path to cascade classifier XML file
cascPath = os.path.dirname(cv2.__file__) + '/data/haarcascade_frontalface_default.xml'
haarcascade = cv2.CascadeClassifier(cascPath)

while True:
    rval, im = cap.read()
    im = cv2.flip(im, 1, 1)

    # Convert RGB to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = haarcascade.detectMultiScale(gray, 
                                         scaleFactor=1.05, 
                                         minNeighbors=5, 
                                         minSize=(40,40), 
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    
    for face in faces:
        # The coordinates of the face
        x, y, w, h = face
        
        # Check the coordinates when face is detected nearly to 4 edges of the frame
        y_new = 0 if y - 40 < 0 else y - 40
        x_new = 0 if x - 40 < 0 else x - 40
        y_h_new = im.shape[0] if y + h + 40 > im.shape[0] else y + h + 40
        x_w_new = im.shape[1] if x + w + 40 > im.shape[1] else x + w + 40
        
        # Area with face
        face_im = im[y_new:y_h_new, x_new:x_w_new]
        
        # Resize frame to 224x224 and normalize
        resized = cv2.resize(face_im, (224,224))
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
