# Face Recognition and Mood Detection WebApp

This repository contains a web application developed using the [Flask-SocketIO](http://flask-socketio.readthedocs.io/en/latest/) web framework to ease the development of a web app using the SocketIO communication protocol with a Python backend and the frontend.

To perform the face detection task, I've used the [Face Detection](https://github.com/ageitgey/face_recognition) library.
For Mood Detection, I've used [Keras](https://keras.io/) with a [Tensorflow](https://www.tensorflow.org/) backend trained using the [Emotion Detection dataset](https://www.kaggle.com/c/emotion-detection-from-facial-expressions) from [Kaggle](https://www.kaggle.com/).

This project uses [Python 3](https://docs.python.org/3/) in the backend for everything.

## Installation

You will need to install OpenCV in your system (to do the video captture and detection of a face). Refer to your system's particular install.

    [Installing OpenCV](https://www.pyimagesearch.com/2018/09/19/pip-install-opencv/)

To set up the development environment install the dependencies:

    pip3 install -r requirements.txt

## Usage

You can install the libraries for this project by running:

```
python3 app.py
```

The script will access your webcam and run the Flask-SocketIO web app on it.
It'll display at ```http://127.0.0.1:5000/``` the name of the person, if recognised, followed by their detected emotion.

## Libraries

    [OpenCV](https://opencv.org/)
    [Keras](https://keras.io/)
    [Tensorflow](https://www.tensorflow.org/)
    [Flask-SocketIO](https://flask-socketio.readthedocs.io/en/latest/)
    [Face Recognition](https://github.com/ageitgey/face_recognition)
    [Kaggle Dataset](https://www.kaggle.com/c/emotion-detection-from-facial-expressions)

## Contributing

1. Fork it
2. Create your feature branch (git checkout -b my-new-feature)
3. Commit your changes (git commit -am 'feat(cool stuff): add some feature')
4. Push to the branch (git push origin my-new-feature)
5. Create a new Pull Request

## Contributors

- [marcosmcb](https://github.com/marcosmcb) - creator, maintainer