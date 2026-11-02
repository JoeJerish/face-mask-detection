/* ==========================================================================
   VISION ASSIST AI — CLIENT-SIDE JAVASCRIPT CONTROLLER & SPEECH ENGINE
   ========================================================================== */

class SpeechManager {
    constructor() {
        self = this;
        this.synth = window.speechSynthesis;
        this.voices = [];
        this.selectedVoice = null;
        this.isEnabled = true;
        this.volume = 1.0;
        this.rate = 1.0; // Normalized 1.0 = ~165 WPM

        this.queue = [];
        this.isSpeaking = false;
        this.lastSpokenTimes = {};
        this.cooldownSeconds = 8;
        this.lastMaskSummary = null;

        // Initialize Web Speech Voices
        this.initVoices();
        if (speechSynthesis.onvoiceschanged !== undefined) {
            speechSynthesis.onvoiceschanged = () => this.initVoices();
        }
    }

    initVoices() {
        if (!this.synth) return;
        this.voices = this.synth.getVoices();
        
        const select = document.getElementById('voice-select');
        if (!select) return;
        select.innerHTML = '';

        let femaleIndex = -1;

        this.voices.forEach((voice, i) => {
            const option = document.createElement('option');
            option.value = i;
            option.textContent = `${voice.name} (${voice.lang})`;
            
            const nameLower = voice.name.toLowerCase();
            if (femaleIndex === -1 && (nameLower.includes('zira') || nameLower.includes('female') || nameLower.includes('eva') || nameLower.includes('samantha') || nameLower.includes('victoria') || nameLower.includes('hazel'))) {
                femaleIndex = i;
                option.textContent += ' ⭐ [Female Preferred]';
            }
            select.appendChild(option);
        });

        if (femaleIndex !== -1) {
            select.selectedIndex = femaleIndex;
            this.selectedVoice = this.voices[femaleIndex];
        } else if (this.voices.length > 0) {
            select.selectedIndex = 0;
            this.selectedVoice = this.voices[0];
        }
    }

    speak(text, priority = 2, force = false) {
        if (!this.isEnabled && !force) return;
        if (!this.synth) return;

        const now = Date.now() / 1000;
        const key = text.toLowerCase().trim();

        if (!force && priority > 1) {
            const lastTime = this.lastSpokenTimes[key] || 0;
            if (now - lastTime < this.cooldownSeconds) return;
        }

        this.lastSpokenTimes[key] = now;

        const utterance = new SpeechSynthesisUtterance(text);
        if (this.selectedVoice) utterance.voice = this.selectedVoice;
        utterance.volume = this.volume;
        utterance.rate = this.rate;

        utterance.onend = () => {
            this.isSpeaking = false;
            this.processQueue();
        };

        utterance.onerror = () => {
            this.isSpeaking = false;
            this.processQueue();
        };

        if (priority === 1) {
            // Cancel current lower priority speech for urgent safety alerts
            this.synth.cancel();
            this.queue = [];
        }

        this.queue.push(utterance);
        this.processQueue();
    }

    processQueue() {
        if (this.isSpeaking || this.queue.length === 0) return;
        this.isSpeaking = true;
        const nextUtterance = this.queue.shift();
        this.synth.speak(nextUtterance);
    }

    processDetectionEvents(maskResults, summaryCounts) {
        if (!this.isEnabled) return;

        // 1. Mask Alerts (Priority 1)
        if (maskResults && maskResults.length > 0) {
            const maskOffCount = maskResults.filter(m => m.status === 'MASK OFF').length;
            const maskOnCount = maskResults.filter(m => m.status === 'MASK ON').length;
            const summaryKey = `${maskOffCount}-${maskOnCount}`;

            if (summaryKey !== this.lastMaskSummary) {
                this.lastMaskSummary = summaryKey;
                if (maskOffCount > 0) {
                    const msg = maskOffCount === 1 ? "Person detected without a mask." : `${maskOffCount} persons detected without masks.`;
                    this.speak(msg, 1, true);
                    logEvent(`⚠️ ALERT: ${msg}`);
                } else if (maskOnCount > 0) {
                    const msg = maskOnCount === 1 ? "Person detected wearing a mask." : `${maskOnCount} persons detected wearing masks.`;
                    this.speak(msg, 2);
                    logEvent(`✅ INFO: ${msg}`);
                }
            }
        }

        // 2. Object Alerts
        const importantObjects = ['Car', 'Bicycle', 'Mobile phone', 'Laptop', 'Bottle', 'Chair', 'Backpack', 'Dog', 'Cat'];
        for (const [objName, count] of Object.entries(summaryCounts)) {
            if (importantObjects.includes(objName)) {
                const msg = count === 1 ? `${objName} detected.` : `${count} ${objName}s detected.`;
                this.speak(msg, 3);
            }
        }
    }
}

// Global App State
let webcamStream = null;
let isRunning = false;
let isPaused = false;
let speechMgr = null;
let aiTimer = null;
let frameCount = 0;
let lastFpsCalc = Date.now();
let confThreshold = 0.40;
let inferenceRateMs = 100;

// DOM Elements
const videoEl = document.getElementById('webcam-video');
const canvasEl = document.getElementById('overlay-canvas');
const ctx = canvasEl.getContext('2d');
const placeholderEl = document.getElementById('video-placeholder');

// Initialize on page load
window.addEventListener('DOMContentLoaded', () => {
    speechMgr = new SpeechManager();
    checkBackendHealth();
});

async function checkBackendHealth() {
    try {
        const res = await fetch('/api/status');
        if (res.ok) {
            const data = await res.json();
            document.getElementById('sys-backend-stat').innerText = 'Online';
            document.getElementById('sys-yolo-stat').innerText = `Loaded (${data.yolo_model})`;
            document.getElementById('sys-mask-stat').innerText = `Loaded (${data.face_mask_model})`;
        }
    } catch (e) {
        document.getElementById('sys-backend-stat').innerText = 'Backend Connection Error';
        document.getElementById('sys-backend-stat').className = 'sys-val text-red';
    }
}

// ---------------- CAMERA CONTROLLER (DUAL ENGINE) ----------------
let telemetryTimer = null;
let useBrowserWebcam = true; // Default to Browser HTML5 Webcam

async function startCamera() {
    if (isRunning) return;

    if (useBrowserWebcam) {
        // Attempt Mode 1: Browser HTML5 getUserMedia
        try {
            const constraints = {
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    frameRate: { ideal: 30 }
                }
            };

            let stream = null;
            try {
                stream = await navigator.mediaDevices.getUserMedia(constraints);
            } catch (e1) {
                stream = await navigator.mediaDevices.getUserMedia({ video: true });
            }

            webcamStream = stream;
            videoEl.srcObject = webcamStream;

            await new Promise((resolve) => {
                videoEl.onloadedmetadata = () => resolve();
            });

            canvasEl.width = videoEl.videoWidth;
            canvasEl.height = videoEl.videoHeight;

            videoEl.style.display = 'block';
            document.getElementById('mjpeg-stream').style.display = 'none';
            placeholderEl.style.display = 'none';

            isRunning = true;
            isPaused = false;

            updateCameraUI(true, 'Browser Webcam (HTML5)', `${videoEl.videoWidth} × ${videoEl.videoHeight}`);
            logEvent('Camera started (Browser HTML5 Webcam).');

            // Start Client Inference & Canvas Render Loops
            startAIInferenceLoop();
            requestAnimationFrame(renderLoop);

            speechMgr.speak("Vision Assist AI is active. Browser camera feed started.", 1, true);
            return;

        } catch (err) {
            console.warn("Browser getUserMedia failed, automatically switching to Backend OpenCV Stream:", err);
            useBrowserWebcam = false; // Fallback to Backend Mode
        }
    }

    // Mode 2: Backend OpenCV Stream (/video_feed)
    try {
        const res = await fetch('/api/camera/start', { method: 'POST' });
        const data = await res.json();

        if (!res.ok || data.error) {
            throw new Error(data.error || 'Failed to start backend camera stream');
        }

        const mjpegImg = document.getElementById('mjpeg-stream');
        mjpegImg.src = '/video_feed?t=' + Date.now();
        mjpegImg.style.display = 'block';
        videoEl.style.display = 'none';

        placeholderEl.style.display = 'none';
        isRunning = true;
        isPaused = false;

        updateCameraUI(true, 'Backend OpenCV Stream', data.resolution || '1280 × 720');
        logEvent('Camera started (Backend OpenCV Stream).');

        // Start Telemetry Polling Loop
        startTelemetryLoop();

        speechMgr.speak("Vision Assist AI is active. Backend camera stream started.", 1, true);

    } catch (err) {
        alert(`Camera Error: ${err.message}\nPlease ensure your camera is connected and permitted.`);
        logEvent(`Camera Error: ${err.message}`);
    }
}

async function stopCamera() {
    isRunning = false;
    if (aiTimer) clearInterval(aiTimer);
    if (telemetryTimer) clearInterval(telemetryTimer);

    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }

    try {
        await fetch('/api/camera/stop', { method: 'POST' });
    } catch (e) {}

    videoEl.srcObject = null;
    videoEl.style.display = 'none';

    const mjpegImg = document.getElementById('mjpeg-stream');
    if (mjpegImg) {
        mjpegImg.src = '';
        mjpegImg.style.display = 'none';
    }

    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    placeholderEl.style.display = 'flex';

    updateCameraUI(false, 'OFFLINE', 'N/A');
    logEvent('Camera stopped.');
}

function updateCameraUI(active, sourceLabel, resText) {
    document.getElementById('btn-start').disabled = active;
    document.getElementById('btn-stop').disabled = !active;
    document.getElementById('btn-pause').disabled = !active;

    if (active) {
        document.getElementById('stat-cam-status').innerText = 'ONLINE';
        document.getElementById('stat-cam-status').className = 'stat-value text-green';
        document.getElementById('sys-cam-stat').innerText = 'Connected';
        document.getElementById('sys-cam-stat').className = 'sys-val text-green';

        document.getElementById('sys-status-badge').className = 'badge badge-online';
        document.getElementById('sys-status-text').innerText = 'AI SYSTEM ONLINE';
        document.getElementById('cam-source-lbl').innerText = sourceLabel;
        document.getElementById('cam-resolution-tag').innerText = `Resolution: ${resText}`;
    } else {
        document.getElementById('stat-cam-status').innerText = 'OFFLINE';
        document.getElementById('stat-cam-status').className = 'stat-value text-red';
        document.getElementById('sys-cam-stat').innerText = 'Disconnected';
        document.getElementById('sys-cam-stat').className = 'sys-val text-red';

        document.getElementById('sys-status-badge').className = 'badge badge-offline';
        document.getElementById('sys-status-text').innerText = 'CAMERA OFFLINE';
        document.getElementById('stat-fps').innerText = '0 FPS';
    }
}

function toggleStreamMode() {
    useBrowserWebcam = !useBrowserWebcam;
    const modeName = useBrowserWebcam ? 'Browser Webcam (HTML5)' : 'Backend OpenCV Stream';
    alert(`Switched stream mode to: ${modeName}\nClick 'START CAMERA' to activate.`);
    logEvent(`Stream mode switched to: ${modeName}`);
    if (isRunning) {
        stopCamera().then(() => startCamera());
    }
}

function startTelemetryLoop() {
    if (telemetryTimer) clearInterval(telemetryTimer);

    telemetryTimer = setInterval(async () => {
        if (!isRunning || isPaused) return;

        try {
            const res = await fetch('/api/live_data');
            if (res.ok) {
                const results = await res.json();
                document.getElementById('stat-fps').innerText = `${Math.round(results.fps || 0)} FPS`;
                document.getElementById('cam-fps-tag').innerText = `${Math.round(results.fps || 0)} FPS`;
                updateSidebar(results);
                speechMgr.processDetectionEvents(results.mask_results, results.summary_counts);
            }
        } catch (e) {
            console.error('Telemetry error:', e);
        }
    }, 200);
}

function togglePause() {
    isPaused = !isPaused;
    const btn = document.getElementById('btn-pause');
    if (isPaused) {
        btn.innerText = '▶ Resume AI';
        logEvent('AI Detection paused.');
    } else {
        btn.innerText = '⏸ Pause AI';
        logEvent('AI Detection resumed.');
    }
}

function toggleFullscreen() {
    const wrapper = document.getElementById('video-wrapper');
    if (!document.fullscreenElement) {
        wrapper.requestFullscreen().catch(err => alert(err.message));
    } else {
        document.exitFullscreen();
    }
}

// ---------------- RENDER & INFERENCE LOOPS ----------------
function renderLoop() {
    if (!isRunning) return;

    // Calculate Client FPS
    frameCount++;
    const now = Date.now();
    if (now - lastFpsCalc >= 1000) {
        const fps = Math.round((frameCount * 1000) / (now - lastFpsCalc));
        document.getElementById('stat-fps').innerText = `${fps} FPS`;
        document.getElementById('cam-fps-tag').innerText = `${fps} FPS`;
        frameCount = 0;
        lastFpsCalc = now;
    }

    requestAnimationFrame(renderLoop);
}

function startAIInferenceLoop() {
    const hiddenCanvas = document.createElement('canvas');
    const hiddenCtx = hiddenCanvas.getContext('2d');

    aiTimer = setInterval(async () => {
        if (!isRunning || isPaused || videoEl.paused || videoEl.ended) return;

        hiddenCanvas.width = videoEl.videoWidth;
        hiddenCanvas.height = videoEl.videoHeight;
        hiddenCtx.drawImage(videoEl, 0, 0);

        const dataUrl = hiddenCanvas.toDataURL('image/jpeg', 0.7);

        try {
            const res = await fetch('/api/detect', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image: dataUrl,
                    conf: confThreshold
                })
            });

            if (res.ok) {
                const results = await res.json();
                drawDetections(results);
                updateSidebar(results);
                speechMgr.processDetectionEvents(results.mask_results, results.summary_counts);
            }
        } catch (e) {
            console.error('Inference error:', e);
        }
    }, inferenceRateMs);
}

// ---------------- BOUNDING BOX OVERLAY RENDERER ----------------
function drawDetections(results) {
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

    const detections = results.detections || [];
    const maskResults = results.mask_results || [];

    // 1. Draw Object Boxes
    detections.forEach(det => {
        const [x1, y1, x2, y2] = det.box;
        const clsName = det.class_name;
        const conf = det.confidence;
        const isPerson = clsName.toLowerCase() === 'person';

        const color = isPerson ? '#06B6D4' : '#F59E0B'; // Cyan for Person, Orange for Objects

        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        const label = `${clsName} ${conf}%`;
        ctx.font = 'bold 13px Outfit, sans-serif';
        const textWidth = ctx.measureText(label).width;

        ctx.fillStyle = color;
        ctx.fillRect(x1, Math.max(0, y1 - 22), textWidth + 10, 22);

        ctx.fillStyle = '#000000';
        ctx.fillText(label, x1 + 5, Math.max(15, y1 - 6));
    });

    // 2. Draw Face & Mask Overlay Boxes
    maskResults.forEach(m => {
        const [fx1, fy1, fx2, fy2] = m.box;
        const status = m.status;
        const conf = m.confidence;

        const isMaskOn = status === 'MASK ON';
        const color = isMaskOn ? '#10B981' : '#EF4444'; // Green for Mask ON, Red for Mask OFF
        const badgeText = `Mask: ${isMaskOn ? 'ON' : 'OFF'} (${conf}%)`;

        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(fx1, fy1, fx2 - fx1, fy2 - fy1);

        ctx.font = 'bold 14px Outfit, sans-serif';
        const textWidth = ctx.measureText(badgeText).width;

        ctx.fillStyle = color;
        ctx.fillRect(fx1, Math.max(0, fy1 - 26), textWidth + 12, 26);

        ctx.fillStyle = '#FFFFFF';
        ctx.fillText(badgeText, fx1 + 6, Math.max(18, fy1 - 7));
    });
}

// ---------------- SIDEBAR & STATS UPDATER ----------------
function updateSidebar(results) {
    const detections = results.detections || [];
    const maskResults = results.mask_results || [];

    document.getElementById('stat-objects-count').innerText = detections.length;
    document.getElementById('stat-persons-count').innerText = results.person_boxes ? results.person_boxes.length : 0;

    // Objects list tags
    const objListEl = document.getElementById('objects-list');
    objListEl.innerHTML = '';
    if (detections.length === 0) {
        objListEl.innerHTML = '<span class="empty-hint">No objects currently detected</span>';
    } else {
        detections.forEach(d => {
            const tag = document.createElement('span');
            tag.className = 'det-tag';
            tag.innerText = `${d.class_name} (${d.confidence}%)`;
            objListEl.appendChild(tag);
        });
    }

    // Mask status cards
    const maskListEl = document.getElementById('mask-status-list');
    maskListEl.innerHTML = '';
    if (maskResults.length === 0) {
        maskListEl.innerHTML = '<div class="empty-hint">No person in frame</div>';
    } else {
        maskResults.forEach((m, i) => {
            const card = document.createElement('div');
            const isMaskOn = m.status === 'MASK ON';
            card.className = `mask-item ${isMaskOn ? 'mask-on' : 'mask-off'}`;
            card.innerHTML = `<span>Person ${i + 1}</span> <span>${m.status} (${m.confidence}%)</span>`;
            maskListEl.appendChild(card);
        });
    }
}

// ---------------- VOICE ASSISTANT INTERACTION ----------------
function updateVoiceSelection() {
    const select = document.getElementById('voice-select');
    const idx = parseInt(select.value);
    if (speechMgr.voices[idx]) {
        speechMgr.selectedVoice = speechMgr.voices[idx];
    }
}

function updateVoiceSettings() {
    const vol = parseFloat(document.getElementById('voice-volume').value);
    const rate = parseInt(document.getElementById('voice-rate').value);

    document.getElementById('vol-val').innerText = Math.round(vol * 100);
    document.getElementById('rate-val').innerText = rate;

    speechMgr.volume = vol;
    speechMgr.rate = rate / 165.0; // Normalize
}

function toggleVoiceAssistant() {
    speechMgr.isEnabled = !speechMgr.isEnabled;
    const btn = document.getElementById('btn-toggle-voice');
    const badge = document.getElementById('voice-active-badge');
    const stat = document.getElementById('stat-voice-status');

    if (speechMgr.isEnabled) {
        btn.innerText = 'Disable Voice';
        btn.className = 'btn btn-primary half';
        badge.innerText = 'ACTIVE';
        badge.className = 'status-badge badge-active';
        stat.innerText = 'ACTIVE';
        stat.className = 'stat-value text-green';
        logEvent('Voice assistant enabled.');
    } else {
        btn.innerText = 'Enable Voice';
        btn.className = 'btn btn-success half';
        badge.innerText = 'DISABLED';
        badge.className = 'status-badge badge-disabled';
        stat.innerText = 'DISABLED';
        stat.className = 'stat-value text-red';
        logEvent('Voice assistant disabled.');
    }
}

function testVoiceAssistant() {
    speechMgr.speak("Vision Assist AI is ready. Camera and voice assistant are active.", 1, true);
}

function logEvent(msg) {
    const box = document.getElementById('event-log-box');
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    const now = new Date().toLocaleTimeString();
    entry.innerHTML = `<span>${now}</span> ${msg}`;
    box.prepend(entry);
}

// Modal Handlers
function switchTab(tabId) {
    const el = document.getElementById(tabId);
    if (el) el.scrollIntoView({ behavior: 'smooth' });
}

function openModal(modalId) {
    document.getElementById(modalId).style.display = 'flex';
}

function closeModal(modalId) {
    document.getElementById(modalId).style.display = 'none';
}

function updateConfThresholdLabel(val) {
    confThreshold = parseFloat(val);
    document.getElementById('conf-val-lbl').innerText = Math.round(confThreshold * 100);
}

function saveSettings() {
    inferenceRateMs = 1000 / parseInt(document.getElementById('inf-rate-slider').value);
    closeModal('settings-modal');
    logEvent('Settings saved successfully.');
}
