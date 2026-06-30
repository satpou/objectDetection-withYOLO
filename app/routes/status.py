from flask import Blueprint, jsonify
from app.services.gesture import MEDIAPIPE_AVAILABLE, landmarker
from app.services.detector import model
from app.services.stream import current_mode

status_bp = Blueprint('status', __name__)

@status_bp.route('/set_mode', methods=['POST'])
def set_mode():
    from flask import request
    global current_mode
    data = request.get_json()
    mode = data.get('mode', 'both')
    if mode in ['yolo', 'hand', 'both']:
        current_mode = mode
        return jsonify({'status': 'success', 'mode': current_mode})
    return jsonify({'status': 'error', 'message': 'Invalid mode'})

@status_bp.route('/get_status')
def get_status():
    return jsonify({
        'mode': current_mode,
        'mediapipe_available': MEDIAPIPE_AVAILABLE,
        'model_loaded': model is not None,
        'landmarker_loaded': landmarker is not None
    })