import face_recognition
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import tensorflow as tf
import time
from utils.helper_functions import get_frames, faces, recognise_faces, display_results
from keras.models import model_from_json

#-----------------------------
#face expression recognizer initialization
face_cascade = cv2.CascadeClassifier('./utils/models/haarcascade_frontalface_default.xml')
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
#-----------------------------
model = None
graph = None
# def get_model():
#     global model
#     if model == None:
        
#         model.load_weights('./utils/models/facial_expression_model_weights.h5') #load weights
#     return model
    
def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    model = model_from_json(open("./utils/models/facial_expression_model_structure.json", "r").read())      
    model.load_weights('./utils/models/facial_expression_model_weights.h5')  
    global graph
    # save graph after loading ResNet50 weights
    graph = tf.get_default_graph()



def get_emotions(gray, img):
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
            
            
            emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
            predicted_emotion = emotions[max_index]
            return predicted_emotion



def detect(socketio):
    load_model()
    # # Load a second sample picture and learn how to recognize it.
    marcos_image = face_recognition.load_image_file("./utils/images/marcos_2.png")
    marcos_face_encoding = face_recognition.face_encodings(marcos_image)[0]

    # Create arrays of known face encodings and their names
    known_face_encodings = [ marcos_face_encoding ]
    known_face_names = [ "Marcos" ]

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)
    while True:
        
        frame, rgb_small_frame, gray_frame = get_frames(video_capture)   
        if process_this_frame:
            face_locations, face_encodings = faces(rgb_small_frame, frame)
            face_names = recognise_faces(face_encodings, known_face_encodings, known_face_names)
            emotion = get_emotions(gray_frame, frame)
            emotion = emotion if emotion != None else " "
        process_this_frame = not process_this_frame

        display_results(frame, face_locations, face_names, emotion, socketio)
        socketio.sleep(1)
        # # Display the resulting image
        # cv2.imshow('Video', frame)

        # # Hit 'q' on the keyboard to quit!
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()