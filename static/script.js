document.addEventListener('DOMContentLoaded', function() {
    const modeBtns = document.querySelectorAll('.mode-btn');
    
    updateStatus();
    
    modeBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            const mode = this.dataset.mode;
            setMode(mode);
        });
    });
    
    setInterval(updateStatus, 2000);
});

function setMode(mode) {
    fetch('/set_mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: mode })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            document.querySelectorAll('.mode-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            document.querySelector(`[data-mode="${mode}"]`).classList.add('active');
            updateStatus();
            showNotification(`Mode: ${mode.toUpperCase()}`, 'success');
        } else {
            showNotification('Error: ' + data.message, 'danger');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showNotification('Failed to change mode', 'danger');
    });
}

function updateStatus() {
    fetch('/get_status')
    .then(response => response.json())
    .then(data => {
        const modeLabels = { both: 'Both', yolo: 'YOLO', hand: 'Hand' };
        const modeEl = document.getElementById('current-mode');
        modeEl.textContent = modeLabels[data.mode] || data.mode;
        
        const colors = { both: 'primary', yolo: 'success', hand: 'warning' };
        modeEl.className = `badge bg-${colors[data.mode] || 'primary'}`;
        
        const modelStatus = document.getElementById('model-status');
        if (data.model_loaded) {
            modelStatus.innerHTML = '✓ Ready';
            modelStatus.className = 'badge bg-success';
        } else {
            modelStatus.innerHTML = '⏳ Loading';
            modelStatus.className = 'badge bg-warning';
        }
        
        const mpStatus = document.getElementById('mediapipe-status');
        if (data.mediapipe_available) {
            if (data.landmarker_loaded) {
                mpStatus.innerHTML = '✓ Ready';
                mpStatus.className = 'badge bg-success';
            } else {
                mpStatus.innerHTML = '⏳ Loading';
                mpStatus.className = 'badge bg-warning';
            }
        } else {
            mpStatus.innerHTML = '✗ Unavailable';
            mpStatus.className = 'badge bg-danger';
        }
    })
    .catch(error => console.error('Error fetching status:', error));
}

function showNotification(message, type) {
    const existing = document.querySelectorAll('.alert');
    existing.forEach(el => el.remove());
    
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.role = 'alert';
    alertDiv.innerHTML = `
        <i class="bi bi-${type === 'success' ? 'check-circle' : 'exclamation-triangle'} me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    const container = document.querySelector('.container-fluid');
    container.insertBefore(alertDiv, container.firstChild);
    
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.classList.remove('show');
            setTimeout(() => alertDiv.remove(), 150);
        }
    }, 3000);
}