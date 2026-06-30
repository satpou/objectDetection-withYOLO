import cv2
import time
import platform

camera = None

def init_camera():
    global camera
    if camera is None:
        if platform.system() == "Darwin":
            camera = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        else:
            camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera

def read_frame():
    global camera
    if camera is None:
        init_camera()
    success, frame = camera.read()
    if not success:
        time.sleep(0.01)
        return read_frame()
    return cv2.flip(frame, 1)

def release():
    global camera
    if camera is not None:
        camera.release()
        camera = None