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
    initializeMetricsView();
    
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
    
    // Load metrics when switching to metrics view
    if (viewName === 'metrics') {
        loadMetrics();
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
            `, antispoofingData);
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
        `, antispoofingData);
        
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

function showResult(type, title, content, antispoofingData = null) {
    const resultBox = document.getElementById('identify-result');
    resultBox.className = `result-box ${type}`;
    
    const icons = {
        success: 'fa-check-circle',
        error: 'fa-times-circle',
        warning: 'fa-exclamation-triangle',
        info: 'fa-info-circle'
    };
    
    // Add feedback buttons if antispoofing data is provided
    let feedbackSection = '';
    if (antispoofingData) {
        feedbackSection = `
            <div class="feedback-section" style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid var(--border);">
                <p style="margin-bottom: 0.5rem; font-size: 0.9rem; color: var(--text-secondary);">
                    Was the liveness detection correct?
                </p>
                <div class="feedback-buttons">
                    <button class="btn btn-success btn-sm" onclick="submitAntispoofingFeedback(true, ${JSON.stringify(antispoofingData).replace(/"/g, '&quot;')})">
                        <i class="fas fa-check"></i> Yes, Correct
                    </button>
                    <button class="btn btn-danger btn-sm" onclick="openFeedbackDialog(false, ${JSON.stringify(antispoofingData).replace(/"/g, '&quot;')})">
                        <i class="fas fa-times"></i> No, Incorrect
                    </button>
                </div>
            </div>
        `;
    }
    
    resultBox.innerHTML = `
        <div class="result-header">
            <i class="fas ${icons[type]}"></i>
            <div class="result-title">${title}</div>
        </div>
        <div class="result-content">
            ${content}
            ${feedbackSection}
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

// ============================================================================
// Anti-Spoofing Feedback Functions
// ============================================================================

async function submitAntispoofingFeedback(isCorrect, antispoofingData) {
    try {
        showLoading('Submitting feedback...');
        
        const feedbackData = {
            detection_result: antispoofingData,
            is_correct: isCorrect,
            true_label: null  // Only needed if incorrect
        };
        
        const response = await fetch(`${API_BASE_URL}/antispoofing/feedback`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(feedbackData)
        });
        
        const result = await response.json();
        
        hideLoading();
        
        if (result.success) {
            showToast('Thank you for your feedback!', 'success');
            
            // Hide feedback buttons after submission
            const feedbackSection = document.querySelector('.feedback-section');
            if (feedbackSection) {
                feedbackSection.innerHTML = '<p style="color: var(--success); font-size: 0.9rem;"><i class="fas fa-check"></i> Feedback submitted</p>';
            }
        } else {
            showToast('Failed to submit feedback: ' + result.error, 'error');
        }
    } catch (error) {
        hideLoading();
        console.error('Feedback error:', error);
        showToast('Failed to submit feedback', 'error');
    }
}

function openFeedbackDialog(isCorrect, antispoofingData) {
    const actualLabel = antispoofingData.classification || 'unknown';
    const options = ['real', 'photo', 'video_replay', 'mask'].filter(opt => opt !== actualLabel);
    
    const dialog = document.createElement('div');
    dialog.className = 'modal-overlay';
    dialog.innerHTML = `
        <div class="modal-dialog">
            <div class="modal-header">
                <h3>Correction Needed</h3>
                <button class="modal-close" onclick="this.closest('.modal-overlay').remove()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div class="modal-body">
                <p>The system detected: <strong>${actualLabel}</strong></p>
                <p>What should it have been?</p>
                <div class="feedback-options">
                    ${options.map(opt => `
                        <button class="btn btn-outline feedback-option-btn" data-label="${opt}">
                            ${opt === 'real' ? '✅ Real Face' : 
                              opt === 'photo' ? '📄 Photo' : 
                              opt === 'video_replay' ? '📺 Video Replay' : 
                              '🎭 Mask'}
                        </button>
                    `).join('')}
                </div>
            </div>
        </div>
    `;
    
    document.body.appendChild(dialog);
    
    // Add click handlers to option buttons
    dialog.querySelectorAll('.feedback-option-btn').forEach(btn => {
        btn.addEventListener('click', async () => {
            const trueLabel = btn.dataset.label;
            dialog.remove();
            
            try {
                showLoading('Submitting correction and adjusting system...');
                
                const feedbackData = {
                    detection_result: antispoofingData,
                    is_correct: false,
                    true_label: trueLabel
                };
                
                const response = await fetch(`${API_BASE_URL}/antispoofing/feedback`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(feedbackData)
                });
                
                const result = await response.json();
                
                hideLoading();
                
                if (result.success) {
                    let message = 'Thank you! System has been adjusted.';
                    if (result.changes_made && result.changes_made.length > 0) {
                        message += ` (${result.changes_made.length} parameters updated)`;
                    }
                    showToast(message, 'success');
                    
                    // Hide feedback buttons after submission
                    const feedbackSection = document.querySelector('.feedback-section');
                    if (feedbackSection) {
                        feedbackSection.innerHTML = '<p style="color: var(--success); font-size: 0.9rem;"><i class="fas fa-check"></i> Feedback submitted and system adjusted</p>';
                    }
                } else {
                    showToast('Failed to submit feedback: ' + result.error, 'error');
                }
            } catch (error) {
                hideLoading();
                console.error('Feedback error:', error);
                showToast('Failed to submit feedback', 'error');
            }
        });
    });
}

// ============================================================================
// Metrics View
// ============================================================================

let metricsCharts = {};

function initializeMetricsView() {
    const refreshBtn = document.getElementById('refresh-metrics');
    const exportBtn = document.getElementById('export-metrics');
    
    refreshBtn.addEventListener('click', loadMetrics);
    exportBtn.addEventListener('click', exportMetrics);
}

async function loadMetrics() {
    console.log('Loading metrics...');
    showLoading('Loading system metrics...');
    
    try {
        const response = await fetch(`${API_BASE_URL}/metrics`);
        const data = await response.json();
        
        console.log('Metrics response:', data);
        
        if (data.success && data.metrics) {
            displayMetrics(data.metrics);
            showToast('Metrics loaded successfully', 'success');
        } else {
            console.log('No metrics available, using demo data');
            showToast('No metrics data available yet. Using demo data.', 'info');
            displayDemoMetrics();
        }
    } catch (error) {
        console.error('Metrics error:', error);
        showToast('Error loading metrics. Using demo data.', 'info');
        displayDemoMetrics();
    } finally {
        hideLoading();
    }
}

function displayMetrics(metrics) {
    console.log('Displaying metrics:', metrics);
    
    // Update KPIs with proper formatting
    document.getElementById('kpi-accuracy').textContent = 
        (metrics.accuracy * 100).toFixed(2) + '%';
    document.getElementById('kpi-far').textContent = 
        (metrics.far * 100).toFixed(3) + '%';
    document.getElementById('kpi-frr').textContent = 
        (metrics.frr * 100).toFixed(3) + '%';
    document.getElementById('kpi-eer').textContent = 
        (metrics.eer * 100).toFixed(3) + '%';
    
    // Create charts with proper error handling
    try {
        if (metrics.roc_data) createROCChart(metrics.roc_data);
    } catch (e) {
        console.error('Error creating ROC chart:', e);
    }
    
    try {
        if (metrics.det_data) createDETChart(metrics.det_data);
    } catch (e) {
        console.error('Error creating DET chart:', e);
    }
    
    try {
        if (metrics.cmc_data) createCMCChart(metrics.cmc_data);
    } catch (e) {
        console.error('Error creating CMC chart:', e);
    }
    
    try {
        if (metrics.confusion_matrix) createConfusionMatrix(metrics.confusion_matrix);
    } catch (e) {
        console.error('Error creating confusion matrix:', e);
    }
    
    try {
        if (metrics.score_distribution) createScoreDistribution(metrics.score_distribution);
    } catch (e) {
        console.error('Error creating score distribution:', e);
    }
    
    try {
        if (metrics.antispoofing_data) createAntiSpoofingChart(metrics.antispoofing_data);
    } catch (e) {
        console.error('Error creating antispoofing chart:', e);
    }
    
    // Populate detailed metrics table
    try {
        if (metrics.detailed_metrics) populateMetricsTable(metrics.detailed_metrics);
    } catch (e) {
        console.error('Error populating metrics table:', e);
    }
}

function displayDemoMetrics() {
    console.log('Displaying demo metrics...');
    
    // Demo data for demonstration purposes
    const demoMetrics = {
        accuracy: 0.967,
        far: 0.015,
        frr: 0.018,
        eer: 0.0165,
        roc_data: generateDemoROCData(),
        det_data: generateDemoDETData(),
        cmc_data: generateDemoCMCData(),
        confusion_matrix: generateDemoConfusionMatrix(),
        score_distribution: generateDemoScoreDistribution(),
        antispoofing_data: generateDemoAntiSpoofingData(),
        detailed_metrics: generateDemoDetailedMetrics()
    };
    
    console.log('Demo metrics data:', demoMetrics);
    displayMetrics(demoMetrics);
}

// Chart creation functions
function createROCChart(data) {
    const ctx = document.getElementById('roc-chart').getContext('2d');
    
    // Destroy existing chart if any
    if (metricsCharts.roc) {
        metricsCharts.roc.destroy();
    }
    
    metricsCharts.roc = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [{
                label: 'ROC Curve',
                data: data.points,
                borderColor: 'rgb(79, 70, 229)',
                backgroundColor: 'rgba(79, 70, 229, 0.1)',
                fill: true,
                tension: 0.4,
                pointRadius: 0,
                borderWidth: 3
            }, {
                label: 'Random Classifier',
                data: [{x: 0, y: 0}, {x: 1, y: 1}],
                borderColor: 'rgb(107, 114, 128)',
                borderDash: [5, 5],
                borderWidth: 2,
                pointRadius: 0,
                fill: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                title: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `TPR: ${context.parsed.y.toFixed(3)}, FPR: ${context.parsed.x.toFixed(3)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    title: {
                        display: true,
                        text: 'False Positive Rate (FAR)'
                    },
                    min: 0,
                    max: 1
                },
                y: {
                    title: {
                        display: true,
                        text: 'True Positive Rate (1 - FRR)'
                    },
                    min: 0,
                    max: 1
                }
            }
        }
    });
    
    // Update info
    document.getElementById('roc-info').innerHTML = 
        `<strong>AUC:</strong> ${data.auc.toFixed(4)} | <strong>Interpretation:</strong> ${getAUCInterpretation(data.auc)}`;
}

function createDETChart(data) {
    const ctx = document.getElementById('det-chart').getContext('2d');
    
    if (metricsCharts.det) {
        metricsCharts.det.destroy();
    }
    
    metricsCharts.det = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [{
                label: 'DET Curve',
                data: data.points,
                borderColor: 'rgb(239, 68, 68)',
                backgroundColor: 'rgba(239, 68, 68, 0.1)',
                fill: false,
                tension: 0.4,
                pointRadius: 0,
                borderWidth: 3
            }, {
                label: 'EER Point',
                data: [data.eer_point],
                borderColor: 'rgb(16, 185, 129)',
                backgroundColor: 'rgb(16, 185, 129)',
                pointRadius: 8,
                pointStyle: 'star',
                showLine: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `FRR: ${context.parsed.y.toFixed(3)}, FAR: ${context.parsed.x.toFixed(3)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'logarithmic',
                    title: {
                        display: true,
                        text: 'False Acceptance Rate (FAR) - Log Scale'
                    }
                },
                y: {
                    type: 'logarithmic',
                    title: {
                        display: true,
                        text: 'False Rejection Rate (FRR) - Log Scale'
                    }
                }
            }
        }
    });
    
    document.getElementById('det-info').innerHTML = 
        `<strong>EER:</strong> ${(data.eer * 100).toFixed(2)}% | Lower EER indicates better performance`;
}

function createCMCChart(data) {
    const ctx = document.getElementById('cmc-chart').getContext('2d');
    
    if (metricsCharts.cmc) {
        metricsCharts.cmc.destroy();
    }
    
    metricsCharts.cmc = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.ranks,
            datasets: [{
                label: 'Cumulative Match Characteristic',
                data: data.accuracies,
                borderColor: 'rgb(16, 185, 129)',
                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                fill: true,
                tension: 0.3,
                pointRadius: 3,
                borderWidth: 3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Rank'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Identification Rate'
                    },
                    min: 0,
                    max: 1,
                    ticks: {
                        callback: function(value) {
                            return (value * 100) + '%';
                        }
                    }
                }
            }
        }
    });
    
    document.getElementById('cmc-info').innerHTML = 
        `<strong>Rank-1:</strong> ${(data.accuracies[0] * 100).toFixed(2)}% | <strong>Rank-5:</strong> ${(data.accuracies[4] * 100).toFixed(2)}%`;
}

function createConfusionMatrix(data) {
    const ctx = document.getElementById('confusion-matrix-chart').getContext('2d');
    
    if (metricsCharts.confusion) {
        metricsCharts.confusion.destroy();
    }
    
    const matrixData = [
        {x: 'Genuine', y: 'Predicted Genuine', v: data.tp},
        {x: 'Genuine', y: 'Predicted Impostor', v: data.fn},
        {x: 'Impostor', y: 'Predicted Genuine', v: data.fp},
        {x: 'Impostor', y: 'Predicted Impostor', v: data.tn}
    ];
    
    metricsCharts.confusion = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['True Positive', 'False Negative', 'False Positive', 'True Negative'],
            datasets: [{
                label: 'Count',
                data: [data.tp, data.fn, data.fp, data.tn],
                backgroundColor: [
                    'rgba(16, 185, 129, 0.7)',
                    'rgba(239, 68, 68, 0.7)',
                    'rgba(245, 158, 11, 0.7)',
                    'rgba(79, 70, 229, 0.7)'
                ],
                borderColor: [
                    'rgb(16, 185, 129)',
                    'rgb(239, 68, 68)',
                    'rgb(245, 158, 11)',
                    'rgb(79, 70, 229)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Count'
                    }
                }
            }
        }
    });
    
    const total = data.tp + data.tn + data.fp + data.fn;
    const accuracy = ((data.tp + data.tn) / total * 100).toFixed(2);
    document.getElementById('confusion-info').innerHTML = 
        `<strong>Total:</strong> ${total} samples | <strong>Accuracy:</strong> ${accuracy}%`;
}

function createScoreDistribution(data) {
    const ctx = document.getElementById('score-dist-chart').getContext('2d');
    
    if (metricsCharts.scoreDist) {
        metricsCharts.scoreDist.destroy();
    }
    
    metricsCharts.scoreDist = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.bins,
            datasets: [{
                label: 'Genuine Scores',
                data: data.genuine,
                backgroundColor: 'rgba(16, 185, 129, 0.6)',
                borderColor: 'rgb(16, 185, 129)',
                borderWidth: 1
            }, {
                label: 'Impostor Scores',
                data: data.impostor,
                backgroundColor: 'rgba(239, 68, 68, 0.6)',
                borderColor: 'rgb(239, 68, 68)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Similarity Score'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Frequency'
                    },
                    beginAtZero: true
                }
            }
        }
    });
    
    document.getElementById('score-info').innerHTML = 
        `<strong>Separation:</strong> ${data.separation.toFixed(3)} | Better separation indicates clearer distinction between genuine and impostor`;
}

function createAntiSpoofingChart(data) {
    const ctx = document.getElementById('antispoofing-chart').getContext('2d');
    
    if (metricsCharts.antispoofing) {
        metricsCharts.antispoofing.destroy();
    }
    
    metricsCharts.antispoofing = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Correctly Classified Live', 'Correctly Classified Spoof', 'Misclassified'],
            datasets: [{
                data: [data.live_correct, data.spoof_correct, data.misclassified],
                backgroundColor: [
                    'rgba(16, 185, 129, 0.8)',
                    'rgba(79, 70, 229, 0.8)',
                    'rgba(239, 68, 68, 0.8)'
                ],
                borderColor: [
                    'rgb(16, 185, 129)',
                    'rgb(79, 70, 229)',
                    'rgb(239, 68, 68)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = ((context.parsed / total) * 100).toFixed(1);
                            return `${context.label}: ${context.parsed} (${percentage}%)`;
                        }
                    }
                }
            }
        }
    });
    
    document.getElementById('antispoofing-info').innerHTML = 
        `<strong>Accuracy:</strong> ${(data.accuracy * 100).toFixed(2)}% | <strong>APCER:</strong> ${(data.apcer * 100).toFixed(2)}% | <strong>BPCER:</strong> ${(data.bpcer * 100).toFixed(2)}%`;
}

function populateMetricsTable(metrics) {
    const tbody = document.querySelector('#detailed-metrics-table tbody');
    tbody.innerHTML = '';
    
    metrics.forEach(metric => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td><strong>${metric.name}</strong></td>
            <td><span class="metric-value">${metric.value}</span></td>
            <td>${metric.description}</td>
        `;
        tbody.appendChild(row);
    });
}

// Demo data generators
function generateDemoROCData() {
    const points = [];
    for (let i = 0; i <= 100; i++) {
        const fpr = i / 100;
        const tpr = 1 - Math.exp(-5 * fpr) * (1 - fpr);
        points.push({x: fpr, y: Math.min(tpr + Math.random() * 0.02, 1)});
    }
    return {
        points: points,
        auc: 0.987
    };
}

function generateDemoDETData() {
    const points = [];
    const eerPoint = {x: 0.0165, y: 0.0165};
    
    for (let i = -3; i <= 0; i += 0.1) {
        const far = Math.pow(10, i);
        const frr = far * (1 + Math.random() * 0.2);
        points.push({x: far, y: frr});
    }
    
    return {
        points: points,
        eer: 0.0165,
        eer_point: eerPoint
    };
}

function generateDemoCMCData() {
    const ranks = Array.from({length: 20}, (_, i) => i + 1);
    const accuracies = ranks.map(r => {
        return Math.min(0.967 + (1 - 0.967) * (1 - Math.exp(-r / 3)), 1);
    });
    
    return {
        ranks: ranks,
        accuracies: accuracies
    };
}

function generateDemoConfusionMatrix() {
    return {
        tp: 482,
        tn: 495,
        fp: 8,
        fn: 15
    };
}

function generateDemoScoreDistribution() {
    const bins = [];
    const genuine = [];
    const impostor = [];
    
    for (let i = 0; i <= 20; i++) {
        const score = i / 20;
        bins.push(score.toFixed(2));
        
        // Genuine scores (higher similarity)
        const genuineVal = Math.exp(-Math.pow(score - 0.85, 2) / 0.02) * 100;
        genuine.push(genuineVal + Math.random() * 10);
        
        // Impostor scores (lower similarity)
        const impostorVal = Math.exp(-Math.pow(score - 0.35, 2) / 0.03) * 80;
        impostor.push(impostorVal + Math.random() * 8);
    }
    
    return {
        bins: bins,
        genuine: genuine,
        impostor: impostor,
        separation: 2.847
    };
}

function generateDemoAntiSpoofingData() {
    return {
        live_correct: 475,
        spoof_correct: 468,
        misclassified: 57,
        accuracy: 0.943,
        apcer: 0.062,  // Attack Presentation Classification Error Rate
        bpcer: 0.053   // Bona Fide Presentation Classification Error Rate
    };
}

function generateDemoDetailedMetrics() {
    return [
        {name: 'True Acceptance Rate (TAR)', value: '96.70%', description: 'Percentage of genuine attempts correctly accepted'},
        {name: 'True Rejection Rate (TRR)', value: '98.50%', description: 'Percentage of impostor attempts correctly rejected'},
        {name: 'False Acceptance Rate (FAR)', value: '1.50%', description: 'Percentage of impostor attempts incorrectly accepted'},
        {name: 'False Rejection Rate (FRR)', value: '1.80%', description: 'Percentage of genuine attempts incorrectly rejected'},
        {name: 'Equal Error Rate (EER)', value: '1.65%', description: 'Point where FAR equals FRR'},
        {name: 'Precision', value: '98.38%', description: 'Proportion of positive identifications that were correct'},
        {name: 'Recall (Sensitivity)', value: '96.97%', description: 'Proportion of actual positives correctly identified'},
        {name: 'F1-Score', value: '97.67%', description: 'Harmonic mean of precision and recall'},
        {name: 'Specificity', value: '98.42%', description: 'Proportion of actual negatives correctly identified'},
        {name: 'Rank-1 Accuracy', value: '96.70%', description: 'Percentage correct on first match'},
        {name: 'Rank-5 Accuracy', value: '99.20%', description: 'Percentage correct within top 5 matches'},
        {name: 'Mean Reciprocal Rank', value: '0.9784', description: 'Average of reciprocal ranks'},
        {name: 'Anti-Spoofing Accuracy', value: '94.30%', description: 'Accuracy of liveness detection'}
    ];
}

function exportMetrics() {
    showToast('Exporting metrics data...', 'info');
    
    // Collect all metrics data
    const metricsData = {
        kpis: {
            accuracy: document.getElementById('kpi-accuracy').textContent,
            far: document.getElementById('kpi-far').textContent,
            frr: document.getElementById('kpi-frr').textContent,
            eer: document.getElementById('kpi-eer').textContent
        },
        timestamp: new Date().toISOString(),
        system: 'FaceGuard System'
    };
    
    // Create and download JSON file
    const dataStr = JSON.stringify(metricsData, null, 2);
    const dataBlob = new Blob([dataStr], {type: 'application/json'});
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `biometric_metrics_${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(url);
    
    showToast('Metrics exported successfully', 'success');
}

function getAUCInterpretation(auc) {
    if (auc >= 0.9) return 'Excellent discrimination';
    if (auc >= 0.8) return 'Good discrimination';
    if (auc >= 0.7) return 'Acceptable discrimination';
    return 'Poor discrimination';
}