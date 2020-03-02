import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import tensorflow as tf
from keras.models import model_from_json

#-----------------------------
#face expression recognizer initialization
face_cascade = cv2.CascadeClassifier('./utils/models/haarcascade_frontalface_default.xml')
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
#-----------------------------

model = None
graph = None


def load_model():
    """
    Load the pre-trained Keras model (here we are using a model
    Pre-trained on the dataset provided by Kaggle, but you can
    Substitute in your own networks just as easily)
    Save graph after loading the weights
    """
    global model
    model = model_from_json(open("./utils/models/facial_expression_model_structure.json", "r").read())      
    model.load_weights('./utils/models/facial_expression_model_weights.h5')  
    global graph
    graph = tf.get_default_graph()



def detect_emotions(gray, img):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        if w > 130: #trick: ignore small faces
            
            detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
            detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
            detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
            
            img_pixels = img_to_array(detected_face)
            img_pixels = np.expand_dims(img_pixels, axis = 0)
            
            img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
            
            #------------------------------
            with graph.as_default():
                predictions = model.predict(img_pixels)
            # predictions = get_model().predict(img_pixels) #store probabilities of 7 expressions
            max_index = np.argmax(predictions[0])
            
            
            predicted_emotion = emotions[max_index]
            return predicted_emotion