import os
import face_recognition
import numpy as np
from utils.helper_functions import get_facename

IMAGE_DIR = "./utils/images/"

def load_face_arrays():
    """
    Create arrays of known face encodings and their names
    """
    known_face_encodings = []
    known_face_names = []
    
    for filename in os.listdir(IMAGE_DIR):
        image = face_recognition.load_image_file(os.path.join(IMAGE_DIR, filename))
        face_encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(get_facename(filename))
        
    return (known_face_encodings, known_face_names)


def get_faces(rgb_frame, face_locations):
    """
    Find all the faces and face encodings in the current frame of video
    """
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    return (face_locations, face_encodings)


def recognise_faces(face_encodings, known_face_encodings, known_face_names):
    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)
    return face_names