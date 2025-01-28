import socketio
import eventlet
import numpy as np
from PIL.ImageFilter import GaussianBlur
from flask import Flask

import base64
from io import BytesIO
from PIL import Image
import cv2
from keras.src.models import model
from tensorflow.python.keras.saving.save import load_model

sio = socketio.Server()
app = Flask(__name__)
speed_limit = 10

def image_process(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img


@sio.on('connect')
def connect(sid, environ):
    print('connected')
    send_control(0,0)

def send_control(steering_angle, throttle):
    sio.emit('throttle', data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__(),
    })

@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = image_process(image)
    image = np.array([image])
    steering_angle = float(model.predict(image))
    throttle = 1.0 - speed / speed_limit
    print('{} {} {}'.format(steering_angle, throttle, speed))
    send_control(steering_angle, throttle)

if __name__ == '__main__':
    model = load_model('model.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)