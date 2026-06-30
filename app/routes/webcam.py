from flask import Blueprint, Response, request, jsonify
from app.services.stream import generate_frames
from app.services.stream import current_mode, blur_active
from app.services.detector import init_model

webcam_bp = Blueprint('webcam', __name__)

@webcam_bp.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')