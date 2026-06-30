// ── Image Upload ──
const imageInput = document.getElementById('imageInput');
const imageDropzone = document.getElementById('imageDropzone');
const imagePreview = document.getElementById('imagePreview');
const previewImg = document.getElementById('previewImg');
const imageFilename = document.getElementById('imageFilename');
const processImageBtn = document.getElementById('processImageBtn');

let selectedImageFile = null;

// Dropzone drag events
['dragenter', 'dragover'].forEach(evt => {
    imageDropzone.addEventListener(evt, e => {
        e.preventDefault();
        imageDropzone.classList.add('dragover');
    });
});

['dragleave', 'drop'].forEach(evt => {
    imageDropzone.addEventListener(evt, e => {
        e.preventDefault();
        imageDropzone.classList.remove('dragover');
    });
});

imageDropzone.addEventListener('drop', e => {
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('image/')) {
        handleImageSelect(files[0]);
    }
});

imageInput.addEventListener('change', e => {
    if (e.target.files.length > 0) {
        handleImageSelect(e.target.files[0]);
    }
});

function handleImageSelect(file) {
    selectedImageFile = file;
    const reader = new FileReader();
    reader.onload = e => {
        previewImg.src = e.target.result;
        imageFilename.textContent = file.name;
        imagePreview.classList.remove('d-none');
        processImageBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

processImageBtn.addEventListener('click', async () => {
    if (!selectedImageFile) return;
    
    processImageBtn.disabled = true;
    processImageBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Processing...';
    
    const formData = new FormData();
    formData.append('file', selectedImageFile);
    
    try {
        const response = await fetch('/upload/image', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            document.getElementById('resultImage').src = `/uploads/${data.filename}`;
            document.getElementById('imageResultText').textContent = data.message;
            document.getElementById('imageResultPlaceholder').classList.add('d-none');
            document.getElementById('imageResult').classList.remove('d-none');
            document.getElementById('downloadImageLink').href = `/uploads/${data.filename}`;
            document.getElementById('downloadImageLink').download = data.filename;
            showNotification(`Success! ${data.message}`, 'success');
        } else {
            showNotification(`Error: ${data.message}`, 'danger');
        }
    } catch (error) {
        showNotification('Failed to process image', 'danger');
    }
    
    processImageBtn.disabled = false;
    processImageBtn.innerHTML = '<i class="bi bi-play-circle me-2"></i>Process Image';
});

// ── Video Upload ──
const videoInput = document.getElementById('videoInput');
const videoDropzone = document.getElementById('videoDropzone');
const videoPreview = document.getElementById('videoPreview');
const videoFilename = document.getElementById('videoFilename');
const processVideoBtn = document.getElementById('processVideoBtn');

let selectedVideoFile = null;

// Dropzone drag events
['dragenter', 'dragover'].forEach(evt => {
    videoDropzone.addEventListener(evt, e => {
        e.preventDefault();
        videoDropzone.classList.add('dragover');
    });
});

['dragleave', 'drop'].forEach(evt => {
    videoDropzone.addEventListener(evt, e => {
        e.preventDefault();
        videoDropzone.classList.remove('dragover');
    });
});

videoDropzone.addEventListener('drop', e => {
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('video/')) {
        handleVideoSelect(files[0]);
    }
});

videoInput.addEventListener('change', e => {
    if (e.target.files.length > 0) {
        handleVideoSelect(e.target.files[0]);
    }
});

function handleVideoSelect(file) {
    selectedVideoFile = file;
    videoFilename.textContent = file.name;
    videoPreview.classList.remove('d-none');
    processVideoBtn.disabled = false;
}

processVideoBtn.addEventListener('click', async () => {
    if (!selectedVideoFile) return;
    
    processVideoBtn.disabled = true;
    processVideoBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Processing...';
    document.getElementById('videoResultPlaceholder').classList.add('d-none');
    document.getElementById('videoResult').classList.add('d-none');
    document.getElementById('videoProcessing').classList.remove('d-none');
    
    const formData = new FormData();
    formData.append('file', selectedVideoFile);
    
    try {
        const response = await fetch('/upload/video', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        document.getElementById('videoProcessing').classList.add('d-none');
        
        if (data.status === 'success') {
            document.getElementById('videoSource').src = `/uploads/${data.filename}`;
            document.getElementById('resultVideo').load();
            document.getElementById('videoResultText').textContent = data.message;
            document.getElementById('videoResult').classList.remove('d-none');
            document.getElementById('downloadVideoLink').href = `/uploads/${data.filename}`;
            document.getElementById('downloadVideoLink').download = data.filename;
            showNotification(`Success! ${data.message}`, 'success');
        } else {
            showNotification(`Error: ${data.message}`, 'danger');
            document.getElementById('videoResultPlaceholder').classList.remove('d-none');
        }
    } catch (error) {
        document.getElementById('videoProcessing').classList.add('d-none');
        showNotification('Failed to process video', 'danger');
        document.getElementById('videoResultPlaceholder').classList.remove('d-none');
    }
    
    processVideoBtn.disabled = false;
    processVideoBtn.innerHTML = '<i class="bi bi-play-circle me-2"></i>Process Video';
});

// ── Show/Hide tabs on page load ──
document.addEventListener('DOMContentLoaded', () => {
    // Make sure webcam video container gets 'loaded' class when video loads
    const videoStream = document.getElementById('video-stream');
    if (videoStream) {
        videoStream.addEventListener('load', () => {
            const container = document.getElementById('videoContainer');
            if (container) container.classList.add('loaded');
        });
    }
});