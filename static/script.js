document.addEventListener('DOMContentLoaded', function() {
    const modeBtns = document.querySelectorAll('.mode-btn');
    
    // Load initial status
    updateStatus();
    
    // Set up mode buttons
    modeBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            const mode = this.dataset.mode;
            setMode(mode);
        });
    });
    
    // Update status every 2 seconds
    setInterval(updateStatus, 2000);
});

function setMode(mode) {
    fetch('/set_mode', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ mode: mode })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            // Update active button
            document.querySelectorAll('.mode-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            document.querySelector(`[data-mode="${mode}"]`).classList.add('active');
            
            // Update status
            updateStatus();
            
            showNotification(`Mode changed to: ${mode.toUpperCase()}`, 'success');
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
        // Update mode badge
        const modeText = data.mode.charAt(0).toUpperCase() + data.mode.slice(1);
        document.getElementById('current-mode').textContent = modeText;
        
        // Update model status
        const modelStatus = document.getElementById('model-status');
        if (data.model_loaded) {
            modelStatus.textContent = '✓ Ready';
            modelStatus.className = 'badge bg-success';
        } else {
            modelStatus.textContent = '⏳ Loading';
            modelStatus.className = 'badge bg-warning';
        }
        
        // Update mediapipe status
        const mpStatus = document.getElementById('mediapipe-status');
        if (data.mediapipe_available) {
            if (data.landmarker_loaded) {
                mpStatus.textContent = '✓ Ready';
                mpStatus.className = 'badge bg-success';
            } else {
                mpStatus.textContent = '⏳ Loading';
                mpStatus.className = 'badge bg-warning';
            }
        } else {
            mpStatus.textContent = '✗ Unavailable';
            mpStatus.className = 'badge bg-danger';
        }
    })
    .catch(error => console.error('Error fetching status:', error));
}

function showNotification(message, type) {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.role = 'alert';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    // Add to top of container
    const container = document.querySelector('.container-fluid');
    container.insertBefore(alertDiv, container.firstChild);
    
    // Auto remove after 3 seconds
    setTimeout(() => {
        alertDiv.remove();
    }, 3000);
}