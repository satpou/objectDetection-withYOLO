from flask import Blueprint, request, jsonify
import uuid
from pathlib import Path
from app.services.detector import process_image, process_video

UPLOAD_FOLDER = Path("static/uploads")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

upload_bp = Blueprint('upload', __name__)

@upload_bp.route('/upload/image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'Empty'}), 400

    import cv2
    img, count = process_image(file.read())
    if img is None:
        return jsonify({'status': 'error', 'message': 'Invalid image'}), 400
    
    filename = f"{uuid.uuid4().hex}_result.jpg"
    cv2.imwrite(str(UPLOAD_FOLDER / filename), img)
    return jsonify({'status': 'success', 'count': count, 'filename': filename, 'message': f'Detected {count} objects'})

@upload_bp.route('/upload/video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'Empty'}), 400

    import os
    input_filename = f"{uuid.uuid4().hex}_input.mp4"
    input_path = UPLOAD_FOLDER / input_filename
    file.save(str(input_path))
    
    output_filename = f"{uuid.uuid4().hex}_result.mp4"
    output_path = UPLOAD_FOLDER / output_filename
    
    frame_count, total_objs = process_video(input_path, output_path)
    
    try:
        os.remove(input_path)
    except:
        pass
    
    return jsonify({
        'status': 'success',
        'frame_count': frame_count,
        'total_objects': total_objs,
        'filename': output_filename,
        'message': f'Processed {frame_count} frames, {total_objs} objects'
    })