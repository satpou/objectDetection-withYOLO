from flask import Blueprint, jsonify
import app.services.stream
from app.services.gesture import MEDIAPIPE_AVAILABLE, init_landmarker
from app.services.detector import init_model

status_bp = Blueprint('status', __name__)

@status_bp.route('/set_mode', methods=['POST'])
def set_mode():
    from flask import request
    data = request.get_json()
    mode = data.get('mode', 'both')
    if mode in ['yolo', 'hand', 'both']:
        app.services.stream.current_mode = mode
        return jsonify({'status': 'success', 'mode': mode})
    return jsonify({'status': 'error', 'message': 'Invalid mode'})

@status_bp.route('/get_status')
def get_status():
    model, _ = init_model()
    landmarker = init_landmarker()
    return jsonify({
        'mode': app.services.stream.current_mode,
        'mediapipe_available': MEDIAPIPE_AVAILABLE,
        'model_loaded': model is not None,
        'landmarker_loaded': landmarker is not None
    })