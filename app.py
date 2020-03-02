from threading import Lock
from flask import Flask, render_template, session, request, copy_current_request_context
from flask_socketio import SocketIO, emit, join_room, leave_room, close_room, rooms, disconnect
from face_emotion_detection import detect


async_mode = None
app = Flask(__name__)
app.config['SECRET_KEY'] = "super-secret$523423"
socketio = SocketIO(app, async_mode=async_mode)
thread = None
thread_lock = Lock()


def background_thread():
    detect(socketio)

@app.route('/')
def index():
    return render_template('index.html', async_mode=socketio.async_mode)

@socketio.on('connect', namespace='/studio')
def test_connect():
    global thread
    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(background_thread)
    emit('response', {'data': 'Connected'})


if __name__ == '__main__':
    socketio.run(app)