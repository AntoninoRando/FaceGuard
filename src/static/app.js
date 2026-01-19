// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// Global State
let registerStream = null;
let identifyStream = null;
let mediaRecorder = null;
let recordedChunks = [];
let recordedBlob = null;
let capturedImage = null;

// Initialize App
document.addEventListener('DOMContentLoaded', () => {
    initializeNavigation();
    initializeRegisterView();
    initializeIdentifyView();
    initializeGalleryView();
    
    // Load gallery on startup
    loadGalleryData();
});

// ============================================================================
// Navigation
// ============================================================================

function initializeNavigation() {
    const navButtons = document.querySelectorAll('.nav-btn');
    
    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const viewName = btn.dataset.view;
            switchView(viewName);
            
            // Update active button
            navButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        });
    });
}

function switchView(viewName) {
    // Hide all views
    document.querySelectorAll('.view').forEach(view => {
        view.classList.remove('active');
    });
    
    // Show selected view
    const targetView = document.getElementById(`${viewName}-view`);
    if (targetView) {
        targetView.classList.add('active');
    }
    
    // Stop any active cameras when switching views
    if (viewName !== 'register') {
        stopRegisterCamera();
    }
    if (viewName !== 'identify') {
        stopIdentifyCamera();
    }
    
    // Reload gallery data when switching to gallery view
    if (viewName === 'gallery') {
        loadGalleryData();
    }
}

// ============================================================================
// Registration View
// ============================================================================

function initializeRegisterView() {
    const startCameraBtn = document.getElementById('start-register-camera');
    const stopCameraBtn = document.getElementById('stop-register-camera');
    const captureBtn = document.getElementById('capture-register');
    const retakeBtn = document.getElementById('retake-register');
    const submitBtn = document.getElementById('submit-register');
    
    startCameraBtn.addEventListener('click', startRegisterCamera);
    stopCameraBtn.addEventListener('click', stopRegisterCamera);
    captureBtn.addEventListener('click', capturePhoto);
    retakeBtn.addEventListener('click', retakePhoto);
    submitBtn.addEventListener('click', submitRegistration);
}

async function startRegisterCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: 'user'
            } 
        });
        
        registerStream = stream;
        const video = document.getElementById('register-video');
        video.srcObject = stream;
        
        // Update UI
        document.getElementById('start-register-camera').style.display = 'none';
        document.getElementById('capture-register').style.display = 'inline-flex';
        document.getElementById('stop-register-camera').style.display = 'inline-flex';
        
        showToast('Camera started successfully', 'success');
    } catch (error) {
        console.error('Camera error:', error);
        showToast('Failed to access camera: ' + error.message, 'error');
    }
}

function stopRegisterCamera() {
    if (registerStream) {
        registerStream.getTracks().forEach(track => track.stop());
        registerStream = null;
        
        const video = document.getElementById('register-video');
        video.srcObject = null;
        
        // Update UI
        document.getElementById('start-register-camera').style.display = 'inline-flex';
        document.getElementById('capture-register').style.display = 'none';
        document.getElementById('stop-register-camera').style.display = 'none';
    }
}

function capturePhoto() {
    const video = document.getElementById('register-video');
    const canvas = document.getElementById('register-canvas');
    const context = canvas.getContext('2d');
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0);
    
    canvas.toBlob(blob => {
        capturedImage = blob;
        
        // Show preview
        const preview = document.getElementById('register-preview');
        const previewImg = document.getElementById('register-preview-img');
        previewImg.src = URL.createObjectURL(blob);
        preview.style.display = 'block';
        
        // Stop camera
        stopRegisterCamera();
        
        showToast('Photo captured successfully', 'success');
    }, 'image/jpeg', 0.95);
}

function retakePhoto() {
    capturedImage = null;
    document.getElementById('register-preview').style.display = 'none';
    document.getElementById('register-result').style.display = 'none';
    startRegisterCamera();
}

async function submitRegistration() {
    const nameInput = document.getElementById('user-name');
    const userName = nameInput.value.trim();
    
    if (!userName) {
        showToast('Please enter your name', 'warning');
        nameInput.focus();
        return;
    }
    
    if (!capturedImage) {
        showToast('Please capture a photo first', 'warning');
        return;
    }
    
    showLoading('Registering user...');
    
    try {
        // Create form data
        const formData = new FormData();
        formData.append('image', capturedImage, 'registration.jpg');
        formData.append('name', userName);
        
        // Submit to register endpoint
        const response = await fetch(`${API_BASE_URL}/register`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        hideLoading();
        
        // Show result
        const resultBox = document.getElementById('register-result');
        resultBox.style.display = 'block';
        
        if (data.success) {
            // Registration successful
            resultBox.className = 'result-box success';
            resultBox.innerHTML = `
                <div class="result-header">
                    <i class="fas fa-check-circle"></i>
                    <div class="result-title">Registration Successful!</div>
                </div>
                <div class="result-content">
                    <p>Welcome, <strong>${data.person_name}</strong>! You have been successfully registered in the system.</p>
                    <div class="result-details" style="margin-top: 1rem;">
                        <div class="result-detail-item">
                            <span class="result-detail-label">Name:</span>
                            <span class="result-detail-value">${data.person_name}</span>
                        </div>
                        <div class="result-detail-item">
                            <span class="result-detail-label">Assigned ID:</span>
                            <span class="result-detail-value">${data.class_id}</span>
                        </div>
                        <div class="result-detail-item">
                            <span class="result-detail-label">Total Users:</span>
                            <span class="result-detail-value">${data.total_classes}</span>
                        </div>
                        <div class="result-detail-item">
                            <span class="result-detail-label">Status:</span>
                            <span class="result-detail-value">✓ Registered</span>
                        </div>
                    </div>
                </div>
            `;
            
            showToast(`Successfully registered ${data.person_name}!`, 'success');
            
            // Clear form
            nameInput.value = '';
            capturedImage = null;
            document.getElementById('register-preview').style.display = 'none';
            
            // Reload gallery data if on gallery view
            loadGalleryData();
        } else {
            // Registration failed
            resultBox.className = 'result-box error';
            resultBox.innerHTML = `
                <div class="result-header">
                    <i class="fas fa-times-circle"></i>
                    <div class="result-title">Registration Failed</div>
                </div>
                <div class="result-content">
                    <p>${data.error || 'Unknown error occurred'}</p>
                </div>
            `;
            
            showToast('Registration failed: ' + (data.error || 'Unknown error'), 'error');
        }
        
    } catch (error) {
        hideLoading();
        console.error('Registration error:', error);
        
        const resultBox = document.getElementById('register-result');
        resultBox.style.display = 'block';
        resultBox.className = 'result-box error';
        resultBox.innerHTML = `
            <div class="result-header">
                <i class="fas fa-times-circle"></i>
                <div class="result-title">Registration Failed</div>
            </div>
            <div class="result-content">
                <p>${error.message}</p>
            </div>
        `;
        
        showToast('Registration failed: ' + error.message, 'error');
    }
}

// ============================================================================
// Identification View
// ============================================================================

function initializeIdentifyView() {
    const startCameraBtn = document.getElementById('start-identify-camera');
    const stopCameraBtn = document.getElementById('stop-identify-camera');
    const startRecordBtn = document.getElementById('start-recording');
    const stopRecordBtn = document.getElementById('stop-recording');
    const retakeBtn = document.getElementById('retake-identify');
    const submitBtn = document.getElementById('submit-identify');
    
    startCameraBtn.addEventListener('click', startIdentifyCamera);
    stopCameraBtn.addEventListener('click', stopIdentifyCamera);
    startRecordBtn.addEventListener('click', startRecording);
    stopRecordBtn.addEventListener('click', stopRecording);
    retakeBtn.addEventListener('click', retakeVideo);
    submitBtn.addEventListener('click', submitIdentification);
}

async function startIdentifyCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: 'user'
            },
            audio: false
        });
        
        identifyStream = stream;
        const video = document.getElementById('identify-video');
        video.srcObject = stream;
        
        // Update UI
        document.getElementById('start-identify-camera').style.display = 'none';
        document.getElementById('start-recording').style.display = 'inline-flex';
        document.getElementById('stop-identify-camera').style.display = 'inline-flex';
        
        showToast('Camera started successfully', 'success');
    } catch (error) {
        console.error('Camera error:', error);
        showToast('Failed to access camera: ' + error.message, 'error');
    }
}

function stopIdentifyCamera() {
    if (identifyStream) {
        identifyStream.getTracks().forEach(track => track.stop());
        identifyStream = null;
        
        const video = document.getElementById('identify-video');
        video.srcObject = null;
        
        // Update UI
        document.getElementById('start-identify-camera').style.display = 'inline-flex';
        document.getElementById('start-recording').style.display = 'none';
        document.getElementById('stop-recording').style.display = 'none';
        document.getElementById('stop-identify-camera').style.display = 'none';
    }
}

function startRecording() {
    recordedChunks = [];
    
    try {
        const video = document.getElementById('identify-video');
        const stream = video.srcObject;
        
        mediaRecorder = new MediaRecorder(stream, {
            mimeType: 'video/webm;codecs=vp9'
        });
        
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                recordedChunks.push(event.data);
            }
        };
        
        mediaRecorder.onstop = () => {
            recordedBlob = new Blob(recordedChunks, { type: 'video/webm' });
            showVideoPreview();
        };
        
        mediaRecorder.start();
        
        // Update UI
        document.getElementById('start-recording').style.display = 'none';
        document.getElementById('stop-recording').style.display = 'inline-flex';
        document.getElementById('recording-indicator').style.display = 'flex';
        
        showToast('Recording started - show your face clearly', 'info');
        
        // Auto-stop after 5 seconds
        setTimeout(() => {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                stopRecording();
            }
        }, 5000);
        
    } catch (error) {
        console.error('Recording error:', error);
        showToast('Failed to start recording: ' + error.message, 'error');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        
        // Update UI
        document.getElementById('recording-indicator').style.display = 'none';
        
        // Stop camera
        stopIdentifyCamera();
        
        showToast('Recording stopped', 'success');
    }
}

function showVideoPreview() {
    const preview = document.getElementById('identify-preview');
    const previewVideo = document.getElementById('identify-preview-video');
    
    previewVideo.src = URL.createObjectURL(recordedBlob);
    preview.style.display = 'block';
}

function retakeVideo() {
    recordedBlob = null;
    recordedChunks = [];
    document.getElementById('identify-preview').style.display = 'none';
    document.getElementById('identify-result').style.display = 'none';
    document.getElementById('identify-progress').style.display = 'none';
    startIdentifyCamera();
}

async function submitIdentification() {
    if (!recordedBlob) {
        showToast('Please record a video first', 'warning');
        return;
    }
    
    // Show progress
    const progressSection = document.getElementById('identify-progress');
    progressSection.style.display = 'block';
    
    const resultBox = document.getElementById('identify-result');
    resultBox.style.display = 'none';
    
    try {
        // Step 1: Anti-Spoofing Detection on Video
        updateStepStatus('antispoofing', 'processing');
        showLoading('Analyzing video for liveness...');
        
        const antispoofingData = await checkAntiSpoofing(recordedBlob);
        
        hideLoading();
        
        if (!antispoofingData.success) {
            updateStepStatus('antispoofing', 'error');
            throw new Error('Anti-spoofing check failed: ' + (antispoofingData.error || 'Unknown error'));
        }
        
        if (!antispoofingData.is_live) {
            updateStepStatus('antispoofing', 'error');
            showResult('error', 'Spoofing Detected!', `
                <p>The system has detected a presentation attack.</p>
                <p><strong>Spoof Score:</strong> ${(antispoofingData.spoof_score * 100).toFixed(1)}%</p>
                <p><strong>Confidence:</strong> ${(antispoofingData.confidence * 100).toFixed(1)}%</p>
                <p><em>Please use a live face, not a photo or video.</em></p>
            `);
            return;
        }
        
        updateStepStatus('antispoofing', 'success', `Live face detected (${(antispoofingData.confidence * 100).toFixed(1)}% confidence)`);
        showToast('Liveness verified ✓', 'success');
        
        // Step 2: Extract frame from video for identification
        showToast('Extracting frame for identification...', 'info');
        const frameBlob = await extractFrameFromVideo(recordedBlob);
        
        // Step 3: Identification
        updateStepStatus('identification', 'processing');
        showLoading('Identifying person...');
        
        const identificationData = await identifyPerson(frameBlob);
        
        hideLoading();
        
        if (!identificationData.success) {
            updateStepStatus('identification', 'error');
            throw new Error('Identification failed: ' + (identificationData.error || 'Unknown error'));
        }
        
        if (identificationData.rejected) {
            updateStepStatus('identification', 'error');
            showResult('warning', 'Unknown Person', `
                <p>You are not registered in the system.</p>
                <p><strong>Minimum Distance:</strong> ${identificationData.min_distance.toFixed(4)}</p>
                <p><strong>Threshold:</strong> ${identificationData.threshold_used.toFixed(4)}</p>
                <p><em>Please register first or try again with better lighting.</em></p>
            `);
            return;
        }
        
        updateStepStatus('identification', 'success', `Identified as ${identificationData.predicted_name}`);
        
        // Show success result
        const topMatches = identificationData.top_matches.slice(0, 3);
        const matchesHtml = topMatches.map(match => `
            <div class="result-detail-item">
                <span class="result-detail-label">#${match.rank} ${match.name}:</span>
                <span class="result-detail-value">
                    Distance: ${match.distance.toFixed(4)} | 
                    Confidence: ${(match.confidence * 100).toFixed(1)}%
                </span>
            </div>
        `).join('');
        
        showResult('success', `Welcome, ${identificationData.predicted_name}!`, `
            <p>You have been successfully identified!</p>
            <div class="result-details" style="margin-top: 1rem;">
                <div class="result-detail-item">
                    <span class="result-detail-label">Identity:</span>
                    <span class="result-detail-value">${identificationData.predicted_name} (ID: ${identificationData.predicted_label})</span>
                </div>
                <div class="result-detail-item">
                    <span class="result-detail-label">Distance:</span>
                    <span class="result-detail-value">${identificationData.min_distance.toFixed(4)}</span>
                </div>
                <div class="result-detail-item">
                    <span class="result-detail-label">Confidence:</span>
                    <span class="result-detail-value">
                        ${(topMatches[0].confidence * 100).toFixed(1)}%
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${topMatches[0].confidence * 100}%"></div>
                        </div>
                    </span>
                </div>
                <div class="result-detail-item">
                    <span class="result-detail-label">Liveness:</span>
                    <span class="result-detail-value">✓ Verified (${(antispoofingData.confidence * 100).toFixed(1)}%)</span>
                </div>
            </div>
            <div class="result-details" style="margin-top: 1rem;">
                <h4 style="margin-bottom: 0.5rem;">Top Matches:</h4>
                ${matchesHtml}
            </div>
        `);
        
        showToast(`Welcome, ${identificationData.predicted_name}!`, 'success');
        
    } catch (error) {
        hideLoading();
        console.error('Identification error:', error);
        showResult('error', 'Identification Failed', `
            <p>${error.message}</p>
            <p><em>Please try again or contact support.</em></p>
        `);
        showToast('Identification failed: ' + error.message, 'error');
    }
}

async function extractFrameFromVideo(videoBlob) {
    return new Promise((resolve, reject) => {
        const video = document.createElement('video');
        video.src = URL.createObjectURL(videoBlob);
        
        video.onloadeddata = () => {
            // Seek to middle of video
            video.currentTime = video.duration / 2;
        };
        
        video.onseeked = () => {
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            
            const context = canvas.getContext('2d');
            context.drawImage(video, 0, 0);
            
            canvas.toBlob(blob => {
                URL.revokeObjectURL(video.src);
                resolve(blob);
            }, 'image/jpeg', 0.95);
        };
        
        video.onerror = () => {
            URL.revokeObjectURL(video.src);
            reject(new Error('Failed to extract frame from video'));
        };
    });
}

async function checkAntiSpoofing(videoBlob) {
    const formData = new FormData();
    // Send as video file with .webm extension (MediaRecorder default format)
    formData.append('file', videoBlob, 'recording.webm');
    
    const response = await fetch(`${API_BASE_URL}/antispoofing/detect`, {
        method: 'POST',
        body: formData
    });
    
    if (!response.ok) {
        const errorText = await response.text();
        console.error('Anti-spoofing error:', response.status, errorText);
        throw new Error(`HTTP ${response.status}: ${errorText || response.statusText}`);
    }
    
    return await response.json();
}

async function identifyPerson(imageBlob) {
    const formData = new FormData();
    formData.append('image', imageBlob, 'frame.jpg');
    
    const response = await fetch(`${API_BASE_URL}/identify?top_k=5`, {
        method: 'POST',
        body: formData
    });
    
    if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    return await response.json();
}

function updateStepStatus(step, status, message = '') {
    const stepElement = document.querySelector(`.step[data-step="${step}"]`);
    if (!stepElement) return;
    
    // Remove all status classes
    stepElement.classList.remove('processing', 'success', 'error');
    
    // Add new status class
    if (status !== 'idle') {
        stepElement.classList.add(status);
    }
    
    // Update status message
    const statusElement = stepElement.querySelector('.step-status');
    if (statusElement) {
        statusElement.textContent = message;
    }
}

function showResult(type, title, content) {
    const resultBox = document.getElementById('identify-result');
    resultBox.className = `result-box ${type}`;
    
    const icons = {
        success: 'fa-check-circle',
        error: 'fa-times-circle',
        warning: 'fa-exclamation-triangle',
        info: 'fa-info-circle'
    };
    
    resultBox.innerHTML = `
        <div class="result-header">
            <i class="fas ${icons[type]}"></i>
            <div class="result-title">${title}</div>
        </div>
        <div class="result-content">
            ${content}
        </div>
    `;
    
    resultBox.style.display = 'block';
}

// ============================================================================
// Gallery View
// ============================================================================

function initializeGalleryView() {
    const searchInput = document.getElementById('search-users');
    searchInput.addEventListener('input', filterUsers);
}

async function loadGalleryData() {
    try {
        // Load health info
        const healthResponse = await fetch(`${API_BASE_URL}/health`);
        const healthData = await healthResponse.json();
        
        document.getElementById('system-status').textContent = healthData.status;
        
        // Load gallery info
        const galleryResponse = await fetch(`${API_BASE_URL}/gallery/info`);
        const galleryData = await galleryResponse.json();
        
        document.getElementById('total-users').textContent = galleryData.total_identities;
        document.getElementById('total-samples').textContent = galleryData.total_embeddings;
        
        // Load identities list
        const identitiesResponse = await fetch(`${API_BASE_URL}/identities`);
        const identitiesData = await identitiesResponse.json();
        
        displayUsers(identitiesData.identities);
        
    } catch (error) {
        console.error('Gallery error:', error);
        showToast('Failed to load gallery data', 'error');
    }
}

function displayUsers(users) {
    const usersList = document.getElementById('users-list');
    
    if (users.length === 0) {
        usersList.innerHTML = '<p style="text-align: center; color: var(--text-secondary);">No users registered yet.</p>';
        return;
    }
    
    usersList.innerHTML = users.map(user => `
        <div class="user-card" data-name="${user.name.toLowerCase()}" data-id="${user.id}">
            <div class="user-icon">
                <i class="fas fa-user"></i>
            </div>
            <div class="user-info-wrapper">
                <div class="user-name">${user.name}</div>
                <div class="user-info">
                    <span><i class="fas fa-hashtag"></i> ID: ${user.id}</span>
                    <span><i class="fas fa-images"></i> ${user.sample_count} samples</span>
                </div>
            </div>
            <button class="btn-delete" onclick="deleteUser(${user.id}, '${user.name.replace(/'/g, "\\'")}')"
                    title="Delete ${user.name}">
                <i class="fas fa-trash"></i>
            </button>
        </div>
    `).join('');
}

function filterUsers() {
    const searchTerm = document.getElementById('search-users').value.toLowerCase();
    const userCards = document.querySelectorAll('.user-card');
    
    userCards.forEach(card => {
        const name = card.dataset.name;
        if (name.includes(searchTerm)) {
            card.style.display = 'flex';
        } else {
            card.style.display = 'none';
        }
    });
}

async function deleteUser(userId, userName) {
    // Confirm deletion
    if (!confirm(`Are you sure you want to delete ${userName} (ID: ${userId})?\n\nThis action cannot be undone!`)) {
        return;
    }
    
    showLoading(`Deleting ${userName}...`);
    
    try {
        const response = await fetch(`${API_BASE_URL}/unregister/${userId}`, {
            method: 'DELETE'
        });
        
        const data = await response.json();
        
        hideLoading();
        
        if (data.success) {
            showToast(`Successfully deleted ${userName}`, 'success');
            // Reload gallery data
            await loadGalleryData();
        } else {
            showToast(`Failed to delete user: ${data.error}`, 'error');
        }
        
    } catch (error) {
        hideLoading();
        console.error('Delete error:', error);
        showToast('Failed to delete user: ' + error.message, 'error');
    }
}

// ============================================================================
// UI Utilities
// ============================================================================

function showLoading(message = 'Processing...') {
    const overlay = document.getElementById('loading-overlay');
    const text = document.getElementById('loading-text');
    text.textContent = message;
    overlay.style.display = 'flex';
}

function hideLoading() {
    const overlay = document.getElementById('loading-overlay');
    overlay.style.display = 'none';
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    
    const icons = {
        success: 'fa-check-circle',
        error: 'fa-times-circle',
        warning: 'fa-exclamation-triangle',
        info: 'fa-info-circle'
    };
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <i class="fas ${icons[type]}"></i>
        <span class="toast-message">${message}</span>
        <button class="toast-close">&times;</button>
    `;
    
    container.appendChild(toast);
    
    // Close button
    toast.querySelector('.toast-close').addEventListener('click', () => {
        toast.remove();
    });
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        toast.remove();
    }, 5000);
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    stopRegisterCamera();
    stopIdentifyCamera();
});
