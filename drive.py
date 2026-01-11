import socketio
import eventlet
from flask import Flask
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2

sio = socketio.Server(async_mode="eventlet")
app = Flask(__name__)

speed_limit = 30


def img_preprocess(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255.0
    return img


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            "steering_angle": str(steering_angle),
            "throttle": str(throttle),
        },
    )


@sio.on("telemetry")
def telemetry(sid, data):
    # Decode image and force RGB
    image = Image.open(BytesIO(base64.b64decode(data["image"]))).convert("RGB")
    image = np.asarray(image)

    image = img_preprocess(image)
    image = np.expand_dims(image, axis=0).astype(np.float32)

    steering_angle = float(model.predict(image, verbose=0)[0][0])
    steering_angle = float(np.clip(steering_angle, -1.0, 1.0))

    speed = float(data["speed"])

    throttle = 1.0 - speed / speed_limit
    throttle *= 1.0 - min(abs(steering_angle), 1.0)
    throttle = float(np.clip(throttle, 0.0, 1.0))

    send_control(steering_angle, throttle)


@sio.on("connect")
def connect(sid, environ):
    print("Connected", sid)
    send_control(0, 1)


if __name__ == "__main__":
    model = load_model("model2_v4.h5", compile=False)

    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(("", 4567)), app)
