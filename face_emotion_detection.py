import face_recognition
import cv2
import time
from utils.helper_functions import get_frames, display_results
from utils.face_recognition import get_faces, recognise_faces
from utils.emotion_detection import load_model, detect_emotions


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
            face_locations, face_encodings = get_faces(rgb_small_frame, frame)
            face_names = recognise_faces(face_encodings, known_face_encodings, known_face_names)
            emotion = detect_emotions(gray_frame, frame)
            emotion = emotion if emotion != None else " "
        process_this_frame = not process_this_frame

        display_results(frame, face_locations, face_names, emotion, socketio)
        socketio.sleep(1)
        

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()