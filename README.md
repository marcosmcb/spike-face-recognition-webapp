# Face Recognition and Mood Detection Application running as a WebApp

[![Crates.io](https://img.shields.io/crates/l/rustc-serialize.svg?maxAge=2592000)]()

This repository contains a web application developed using the [Flask-SocketIO](http://flask-socketio.readthedocs.io/en/latest/) web framework to ease the development of a web app using the SocketIO communication protocol with a Python backend and the frontend.

To perform the face detection task, I've used the [Face Detection](https://github.com/ageitgey/face_recognition) library.
For Mood Detection, I've used [Keras](https://keras.io/) with a [Tensorflow](https://www.tensorflow.org/) backend trained using the [Emotion Detection dataset](https://www.kaggle.com/c/emotion-detection-from-facial-expressions) from [Kaggle](https://www.kaggle.com/).


## Usage

You can install the libraries for this project by running:

```
pip install -r requirements.txt
```

The script will access your webcam and run thte Flask-SocketIO web app on it.
It'll display at ```http://127.0.0.1:5000/``` the name of the person, if recognised, followed by their detectted emotion.

## License

Copyright 2020 Marcos Cavalcante

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.