import cv2


def get_facename(filename):
    """
    Get name of the face from filename.
    If, for instance, we have the following filenames, the function returns:
    obama.png ----> obama
    obama_1.png --> obama
    obama_2.png --> obama
    """
    filename_without_extension = filename.split(".")[0]
    face_name = filename_without_extension.split("_")[0]
    return face_name

def get_frames(video):
    # Grab a single frame of video
    ret, frame = video.read()
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return (frame, rgb_small_frame, gray_frame)


def display_results(frame, face_locations, face_names, emotion, socketio):
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name + "   " + emotion, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        socketio.emit('response', {'data': name, 'emotion': emotion},  namespace='/studio')