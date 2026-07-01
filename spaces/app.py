import os
import cv2
import uuid
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string, send_from_directory

UPLOAD = Path("static/uploads")
UPLOAD.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('static/uploads', filename)

HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Visatra - Vision + Satria</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
<link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.0/font/bootstrap-icons.css" rel="stylesheet">
<style>
* { box-sizing: border-box; }
body { background: #0d1117; color: #e6edf3; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; min-height: 100vh; display: flex; flex-direction: column; }
.navbar { background: #161b22; border-bottom: 1px solid #30363d; padding: 12px 0; }
.navbar-brand { color: #e6edf3; font-weight: 700; font-size: 1.2rem; }
.nav-tabs { border-bottom: 1px solid #30363d; justify-content: center; }
.nav-tabs .nav-link { color: #8b949e; border: 1px solid transparent; padding: 10px 20px; margin: 0 4px; border-radius: 8px 8px 0 0; }
.nav-tabs .nav-link.active { background: #161b22; color: #58a6ff; border-color: #30363d #30363d #161b22; }
.nav-tabs .nav-link:hover { color: #e6edf3; }
.card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; color: #e6edf3; }
.card-header { background: #0d1117; border-bottom: 1px solid #30363d; padding: 12px 16px; font-weight: 600; }
.card-body { padding: 16px; }
.btn-primary { background: #238636; border: none; }
.btn-primary:hover { background: #2ea043; }
.btn-success { background: #238636; border: none; }
.btn-success:hover { background: #2ea043; }
.btn-danger { background: #da3633; border: none; }
.btn-danger:hover { background: #f85149; }
.btn-mode { background: #21262d; border: 1px solid #30363d; color: #e6edf3; text-align: left; padding: 10px 14px; margin-bottom: 8px; border-radius: 6px; }
.btn-mode:hover { background: #30363d; }
.btn-mode.active { background: #1f6feb33; border-color: #1f6feb; color: #58a6ff; }
.upload-zone { border: 2px dashed #30363d; border-radius: 8px; padding: 40px 20px; text-align: center; cursor: pointer; transition: all 0.3s; }
.upload-zone:hover { border-color: #58a6ff; background: rgba(88,166,255,0.05); }
.badge-success { background: #238636; }
.badge-secondary { background: #30363d; }
.table { color: #e6edf3; }
.table td { border-color: #21262d; }
.footer { background: #161b22; border-top: 1px solid #30363d; padding: 12px 0; text-align: center; font-size: 0.85rem; color: #8b949e; }
.video-container { background: #000; display: flex; align-items: center; justify-content: center; min-height: 300px; border-radius: 0 0 8px 8px; overflow: hidden; }
.video-container img, .video-container video { max-width: 100%; max-height: 500px; }
.result-container { background: #0d1117; border-radius: 8px; min-height: 200px; display: flex; align-items: center; justify-content: center; overflow: hidden; }
.result-container img { max-width: 100%; max-height: 500px; border-radius: 8px; }
.spinner-border { width: 2rem; height: 2rem; }
.toast-container { position: fixed; top: 70px; right: 20px; z-index: 9999; }
.main-content { flex: 1; }
</style>
</head>
<body>

<nav class="navbar">
<div class="container d-flex justify-content-between align-items-center">
<span class="navbar-brand mb-0"><i class="bi bi-camera-video me-2"></i>Visatra</span>
<div class="d-flex align-items-center gap-3">
<span class="text-muted small d-none d-md-inline">v1.1.0</span>
<span class="badge bg-success" id="status-badge"><i class="bi bi-wifi me-1"></i>Ready</span>
</div>
</div>
</nav>

<div class="main-content">
<div class="container py-4">

<ul class="nav nav-tabs mb-4" id="mainTab">
<li class="nav-item">
<button class="nav-link active" onclick="switchTab('webcam')"><i class="bi bi-webcam me-2"></i>Webcam</button>
</li>
<li class="nav-item">
<button class="nav-link" onclick="switchTab('image')"><i class="bi bi-image me-2"></i>Image</button>
</li>
<li class="nav-item">
<button class="nav-link" onclick="switchTab('video')"><i class="bi bi-film me-2"></i>Video</button>
</li>
</ul>

<!-- Webcam Tab -->
<div id="tab-webcam" class="tab-content">
<div class="row g-4 justify-content-center">
<div class="col-lg-9">
<div class="card">
<div class="card-header"><i class="bi bi-broadcast"></i> Live Camera Feed</div>
<div class="video-container">
<div class="text-center p-5" id="webcam-placeholder">
<i class="bi bi-webcam display-1 text-muted mb-3 d-block"></i>
<h5>Webcam requires local deployment</h5>
<p class="text-muted">Cloud servers do not have camera access. Use Image tab to upload photos instead.</p>
<button class="btn btn-primary mt-2" onclick="switchTab('image')"><i class="bi bi-image me-2"></i>Use Image Tab</button>
</div>
</div>
</div>
</div>
<div class="col-lg-3">
<div class="card mb-3">
<div class="card-header"><i class="bi bi-sliders"></i> Detection Mode</div>
<div class="card-body">
<button class="btn btn-mode active" onclick="setMode(this,'both')"><i class="bi bi-stars"></i> Both (YOLO + Gesture)</button>
<button class="btn btn-mode" onclick="setMode(this,'yolo')"><i class="bi bi-box-seam"></i> YOLO Detection</button>
<button class="btn btn-mode" onclick="setMode(this,'hand')"><i class="bi bi-hand-index-thumb"></i> Hand Gesture</button>
</div>
</div>
<div class="card">
<div class="card-header"><i class="bi bi-activity"></i> System Status</div>
<div class="card-body">
<table class="table"><tbody>
<tr><td>Mode:</td><td><span id="current-mode" class="badge bg-primary">Both</span></td></tr>
<tr><td>YOLO:</td><td><span class="badge bg-success">Loaded</span></td></tr>
<tr><td>MediaPipe:</td><td><span class="badge bg-success">Loaded</span></td></tr>
</tbody></table>
</div>
</div>
</div>
</div>
</div>

<!-- Image Tab -->
<div id="tab-image" class="tab-content" style="display:none">
<div class="row justify-content-center">
<div class="col-md-5 col-lg-4 mb-4 mb-md-0">
<div class="card h-100">
<div class="card-header"><i class="bi bi-cloud-upload"></i> Upload Image</div>
<div class="card-body d-flex flex-column">
<div class="upload-zone flex-grow-1" id="image-dropzone">
<i class="bi bi-cloud-arrow-up display-3 text-primary mb-3 d-block"></i>
<h5>Drag & Drop Image</h5>
<p class="text-muted mb-3">or click to browse</p>
<input type="file" id="image-input" accept="image/*" class="d-none">
<button class="btn btn-primary btn-lg" onclick="document.getElementById('image-input').click()"><i class="bi bi-folder2-open me-2"></i>Browse</button>
</div>
<div id="image-preview" class="mt-3 text-center" style="display:none">
<img id="preview-img" class="img-fluid rounded mb-2" style="max-height:200px">
<span id="image-filename" class="badge bg-secondary"></span>
</div>
<button id="process-image-btn" class="btn btn-success btn-lg w-100 mt-3" disabled onclick="processImage()">
<i class="bi bi-play-circle me-2"></i>Process Image
</button>
</div>
</div>
</div>
<div class="col-md-7 col-lg-8">
<div class="card h-100">
<div class="card-header d-flex justify-content-between align-items-center">
<span><i class="bi bi-image"></i> Detection Result</span>
<a id="download-image" class="btn btn-sm btn-outline-light d-none" download><i class="bi bi-download me-1"></i>Download</a>
</div>
<div class="card-body d-flex align-items-center justify-content-center result-container" style="min-height:400px">
<div id="image-placeholder" class="text-center">
<i class="bi bi-image display-1 mb-3 d-block text-muted" style="opacity:0.5"></i>
<h5>Upload an image</h5>
<p class="small text-muted">Detection results will appear here</p>
</div>
<div id="image-result" class="text-center" style="display:none">
<img id="result-image" class="img-fluid rounded">
<div id="image-stats" class="mt-3 p-3 rounded text-center" style="background:rgba(0,0,0,0.3)">
<span id="image-result-text" class="badge bg-success fs-6"></span>
</div>
</div>
</div>
</div>
</div>
</div>
</div>

<!-- Video Tab -->
<div id="tab-video" class="tab-content" style="display:none">
<div class="row justify-content-center">
<div class="col-md-5 col-lg-4 mb-4 mb-md-0">
<div class="card h-100">
<div class="card-header"><i class="bi bi-cloud-arrow-up"></i> Upload Video</div>
<div class="card-body d-flex flex-column">
<div class="upload-zone flex-grow-1" id="video-dropzone">
<i class="bi bi-film display-3 text-warning mb-3 d-block"></i>
<h5>Drag & Drop Video</h5>
<p class="text-muted mb-3">MP4, AVI, MOV (max 100MB)</p>
<input type="file" id="video-input" accept="video/*" class="d-none">
<button class="btn btn-warning btn-lg" onclick="document.getElementById('video-input').click()"><i class="bi bi-folder2-open me-2"></i>Browse</button>
</div>
<div id="video-preview" class="mt-3 text-center" style="display:none">
<i class="bi bi-film text-warning fs-1 d-block mb-2"></i>
<span id="video-filename" class="badge bg-secondary"></span>
</div>
<button id="process-video-btn" class="btn btn-success btn-lg w-100 mt-3" disabled onclick="processVideo()">
<i class="bi bi-play-circle me-2"></i>Process Video
</button>
</div>
</div>
</div>
<div class="col-md-7 col-lg-8">
<div class="card h-100">
<div class="card-header d-flex justify-content-between align-items-center">
<span><i class="bi bi-film"></i> Detection Result</span>
<a id="download-video" class="btn btn-sm btn-outline-light d-none" download><i class="bi bi-download me-1"></i>Download</a>
</div>
<div class="card-body d-flex align-items-center justify-content-center result-container" style="min-height:400px">
<div id="video-placeholder" class="text-center">
<i class="bi bi-film display-1 mb-3 d-block text-muted" style="opacity:0.5"></i>
<h5>Upload a video</h5>
<p class="small text-muted">Detection results will appear here</p>
</div>
<div id="video-processing" class="text-center" style="display:none">
<div class="spinner-border text-primary mb-3" role="status"></div>
<h5>Processing Video...</h5>
<p class="text-muted small" id="video-processing-text">Please wait</p>
</div>
<div id="video-result" class="text-center" style="display:none">
<div class="ratio ratio-16x9 rounded overflow-hidden mb-3">
<video id="result-video" controls class="w-100"></video>
</div>
<div class="p-3 rounded text-center" style="background:rgba(0,0,0,0.3)">
<span id="video-result-text" class="badge bg-success fs-6"></span>
</div>
</div>
</div>
</div>
</div>
</div>
</div>

</div>
</div>

<footer class="footer">
<div class="container">
<span>Visatra v1.1.0</span> |
<span>Powered by YOLOv8 & MediaPipe</span> |
<span style="font-weight:600">Vision + Satria</span>
</div>
</footer>

<div class="toast-container" id="toast-container"></div>

<script>
let currentMode = 'both';

function switchTab(name) {
    document.querySelectorAll('.tab-content').forEach(t => t.style.display = 'none');
    document.querySelectorAll('.nav-link').forEach(n => n.classList.remove('active'));
    document.getElementById('tab-' + name).style.display = 'block';
    event.target.classList.add('active');
}

function setMode(btn, mode) {
    currentMode = mode;
    btn.closest('.card-body').querySelectorAll('.btn-mode').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const labels = {both:'Both',yolo:'YOLO',hand:'Hand'};
    document.getElementById('current-mode').textContent = labels[mode];
}

function showToast(msg, type) {
    const c = document.getElementById('toast-container');
    const t = document.createElement('div');
    t.className = 'alert alert-' + type + ' alert-dismissible fade show';
    t.innerHTML = msg + '<button type="button" class="btn-close btn-close-white" data-bs-dismiss="alert"></button>';
    c.appendChild(t);
    setTimeout(() => t.remove(), 4000);
}

// Image upload
const imageInput = document.getElementById('image-input');
imageInput.addEventListener('change', function() {
    const f = this.files[0];
    if (!f) return;
    const r = new FileReader();
    r.onload = function(e) {
        document.getElementById('preview-img').src = e.target.result;
        document.getElementById('image-filename').textContent = f.name;
        document.getElementById('image-preview').style.display = 'block';
        document.getElementById('process-image-btn').disabled = false;
    };
    r.readAsDataURL(f);
});

// Image drag-drop
const idz = document.getElementById('image-dropzone');
idz.addEventListener('click', () => imageInput.click());
idz.addEventListener('dragover', e => { e.preventDefault(); idz.style.borderColor = '#58a6ff'; });
idz.addEventListener('dragleave', () => { idz.style.borderColor = '#30363d'; });
idz.addEventListener('drop', e => {
    e.preventDefault(); idz.style.borderColor = '#30363d';
    if (e.dataTransfer.files[0]) { imageInput.files = e.dataTransfer.files; imageInput.dispatchEvent(new Event('change')); }
});

function processImage() {
    const f = imageInput.files[0];
    if (!f) return;
    const fd = new FormData();
    fd.append('file', f);
    fd.append('mode', currentMode);
    document.getElementById('image-placeholder').style.display = 'none';
    document.getElementById('image-result').style.display = 'none';
    document.getElementById('process-image-btn').disabled = true;

    fetch('/upload/image', {method:'POST', body:fd})
    .then(r => r.json())
    .then(d => {
        if (d.status === 'success') {
            document.getElementById('result-image').src = '/uploads/' + d.filename;
            document.getElementById('image-result-text').textContent = d.message;
            document.getElementById('image-result').style.display = 'block';
            document.getElementById('download-image').href = '/uploads/' + d.filename;
            document.getElementById('download-image').classList.remove('d-none');
            showToast('Detection complete! ' + d.count + ' objects found.', 'success');
        } else {
            showToast(d.message || 'Error', 'danger');
        }
    })
    .catch(e => showToast('Error: ' + e, 'danger'))
    .finally(() => document.getElementById('process-image-btn').disabled = false);
}

// Video upload
const videoInput = document.getElementById('video-input');
videoInput.addEventListener('change', function() {
    const f = this.files[0];
    if (!f) return;
    document.getElementById('video-filename').textContent = f.name;
    document.getElementById('video-preview').style.display = 'block';
    document.getElementById('process-video-btn').disabled = false;
});

const vdz = document.getElementById('video-dropzone');
vdz.addEventListener('click', () => videoInput.click());
vdz.addEventListener('dragover', e => { e.preventDefault(); vdz.style.borderColor = '#f0883e'; });
vdz.addEventListener('dragleave', () => { vdz.style.borderColor = '#30363d'; });
vdz.addEventListener('drop', e => {
    e.preventDefault(); vdz.style.borderColor = '#30363d';
    if (e.dataTransfer.files[0]) { videoInput.files = e.dataTransfer.files; videoInput.dispatchEvent(new Event('change')); }
});

function processVideo() {
    const f = videoInput.files[0];
    if (!f) return;
    const fd = new FormData();
    fd.append('file', f);
    fd.append('mode', currentMode);
    document.getElementById('video-placeholder').style.display = 'none';
    document.getElementById('video-result').style.display = 'none';
    document.getElementById('video-processing').style.display = 'block';
    document.getElementById('process-video-btn').disabled = true;

    fetch('/upload/video', {method:'POST', body:fd})
    .then(r => r.json())
    .then(d => {
        document.getElementById('video-processing').style.display = 'none';
        if (d.status === 'success') {
            document.getElementById('result-video').src = '/uploads/' + d.filename;
            document.getElementById('video-result-text').textContent = d.message;
            document.getElementById('video-result').style.display = 'block';
            document.getElementById('download-video').href = '/uploads/' + d.filename;
            document.getElementById('download-video').classList.remove('d-none');
            showToast('Video processed! ' + d.total_objects + ' objects in ' + d.frame_count + ' frames.', 'success');
        } else {
            showToast(d.message || 'Error', 'danger');
            document.getElementById('video-placeholder').style.display = 'block';
        }
    })
    .catch(e => { document.getElementById('video-processing').style.display = 'none'; document.getElementById('video-placeholder').style.display = 'block'; showToast('Error: ' + e, 'danger'); })
    .finally(() => document.getElementById('process-video-btn').disabled = false);
}
</script>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>'''

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/upload/image', methods=['POST'])
def upload_image():
    f = request.files.get('file')
    mode = request.form.get('mode', 'both')
    if not f:
        return jsonify({'status':'error','message':'No file'})
    data = np.frombuffer(f.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'status':'error','message':'Invalid image'})

    try:
        from detector import process_frame
        from gesture import detect_gesture, process_with_gesture
        import cv2 as cv

        if mode in ['hand', 'both']:
            lm, gesture = detect_gesture(cv.cvtColor(img, cv.COLOR_BGR2RGB))
            if lm:
                from drawing import draw_hand_landmarks
                h, w = img.shape[:2]
                draw_hand_landmarks(img, lm, w, h)

        if mode in ['yolo', 'both']:
            img, count = process_frame(img)
        else:
            count = 0

        fname = f"{uuid.uuid4().hex}.jpg"
        cv2.imwrite(str(UPLOAD / fname), img)
        return jsonify({'status':'success','count':count,'filename':fname,'message':f'Detected {count} objects'})
    except Exception as e:
        return jsonify({'status':'error','message':str(e)})

@app.route('/upload/video', methods=['POST'])
def upload_video():
    f = request.files.get('file')
    mode = request.form.get('mode', 'both')
    if not f:
        return jsonify({'status':'error','message':'No file'})

    infname = f"{uuid.uuid4().hex}_input.mp4"
    inpath = UPLOAD / infname
    f.save(str(inpath))

    outfname = f"{uuid.uuid4().hex}_output.mp4"
    outpath = UPLOAD / outfname

    try:
        from detector import process_frame, process_video
        frame_count, total_objs = process_video(str(inpath), str(outpath))
        try: os.remove(inpath)
        except: pass
        return jsonify({'status':'success','frame_count':frame_count,'total_objects':total_objs,'filename':outfname,'message':f'{frame_count} frames, {total_objs} objects'})
    except Exception as e:
        try: os.remove(inpath)
        except: pass
        return jsonify({'status':'error','message':str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)