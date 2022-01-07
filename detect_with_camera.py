from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from mtcnn.mtcnn import MTCNN
import numpy as np
from numpy import asarray
from PIL import Image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine
import cv2

def highlight_faces(image, faces):
  
    for face in faces:
        x, y, width, height = face['box']
        cv2.rectangle(image,(x,y),(x + width, y + height),color=(0,0,255),thickness=4)

cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

detector = MTCNN()

while True:
    
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    #detect faces
    faces = detector.detect_faces(frame_rgb) 
    
    #highlight_faces(frame, faces)
    
    for face in faces:
        x, y, width, height = face['box']
        cv2.rectangle(frame, (x,y), (x + width, y + height), color=(0,0,255), thickness=4)
        
    cv2.imshow("frame", frame)
    # press ESC to close window
    if cv2.waitKey(1) & 0xFF == 27:
        break
        
cap.release()
cv2.destroyAllWindows()