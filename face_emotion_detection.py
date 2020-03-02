import face_recognition
import cv2
import time
from utils.helper_functions import get_frames, display_results
from utils.face_recognition import get_faces, recognise_faces, load_face_arrays
from utils.emotion_detection import load_model, detect_emotions


def detect(socketio):
    load_model()
    # Initialize some variables
    known_face_encodings, known_face_names = load_face_arrays()
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
            emotion = emotion if emotion != None else ""
        process_this_frame = not process_this_frame

        display_results(frame, face_locations, face_names, emotion, socketio)
        socketio.sleep(1)
        
    video_capture.release()
    cv2.destroyAllWindows()