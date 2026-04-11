const API_BASE_URL = 'http://127.0.0.1:8000';

document.addEventListener('DOMContentLoaded', () => {
    setupNavigation();
    setupScanner();
    setupVerifier();
    setupSimulator();
});

// --- NAVIGATION ---
function setupNavigation() {
    const navLinks = document.querySelectorAll('.nav-links li');
    const sections = document.querySelectorAll('.page-section');
    const pageTitle = document.getElementById('page-title');

    const titles = {
        'dashboard': 'Dashboard Overview',
        'scanner': 'Deepfake Media Scanner',
        'verifier': 'Blockchain Integrity Verifier',
        'simulator': 'Predictive Threat Simulator'
    };

    navLinks.forEach(link => {
        link.addEventListener('click', () => {
            // Update active link
            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');

            // Show target section
            const targetId = link.getAttribute('data-target');
            sections.forEach(sec => sec.classList.remove('active'));
            document.getElementById(targetId).classList.add('active');

            // Update title
            pageTitle.innerText = titles[targetId] || 'DeepShield AI';
        });
    });
}

// --- UTILITIES ---
function setupDropzone(dropzoneId, fileInputId, handleFileCallback) {
    const dropzone = document.getElementById(dropzoneId);
    const fileInput = document.getElementById(fileInputId);

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Highlight dropzone
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, () => {
            dropzone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, () => {
            dropzone.classList.remove('dragover');
        }, false);
    });

    // Handle dropped files
    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFileCallback(files[0]);
        }
    });

    // Handle browse files
    fileInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            handleFileCallback(this.files[0]);
        }
    });
}

function updateThreatBadge(elementId, value) {
    const el = document.getElementById(elementId);
    el.innerText = value;
    el.className = 'value threat-badge'; // Reset classes
    
    if (value === 'HIGH') el.classList.add('danger');
    else if (value === 'MEDIUM') el.classList.add('warning');
    else el.classList.add('safe');
}

function updateScoreBadge(elementId, score) {
    const el = document.getElementById(elementId);
    el.innerText = (score * 100).toFixed(1) + '%';
    el.className = 'value score-badge';
    
    if (score > 0.7) el.classList.add('danger');
    else if (score > 0.4) el.classList.add('warning');
    else el.classList.add('safe');
}

// --- DASHBOARD STATE ---
let filesScannedCount = 0;
let threatsInterceptedCount = 0;

function updateDashboardMetrics(threatLevel) {
    // Increment files scanned
    filesScannedCount++;
    const filesEl = document.getElementById('stat-files-scanned');
    if (filesEl) filesEl.innerText = filesScannedCount.toLocaleString();

    // Increment threats if applicable
    if (threatLevel === 'HIGH' || threatLevel === 'MEDIUM') {
        threatsInterceptedCount++;
        const threatsEl = document.getElementById('stat-threats-intercepted');
        if (threatsEl) threatsEl.innerText = threatsInterceptedCount.toLocaleString();
    }
}

// --- SCANNER LOGIC ---
function setupScanner() {
    const loading = document.getElementById('scanner-loading');
    const results = document.getElementById('scanner-results');

    setupDropzone('scanner-dropzone', 'scanner-file', async (file) => {
        results.classList.add('hidden');
        loading.classList.remove('hidden');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch(`${API_BASE_URL}/upload-media`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('API Error');
            const data = await response.json();

            // Populate results
            document.getElementById('res-scan-filename').innerText = data.file_name;
            document.getElementById('res-scan-size').innerText = data.file_size_mb;
            document.getElementById('res-scan-hash').innerText = data.sha256_hash;
            updateScoreBadge('res-scan-score', data.fake_score);
            updateThreatBadge('res-scan-threat', data.threat_prediction);

            // Update Dashboard Metrics
            updateDashboardMetrics(data.threat_prediction);

            loading.classList.add('hidden');
            results.classList.remove('hidden');

        } catch (error) {
            console.error(error);
            alert("Analysis failed. Make sure the backend is running.");
            loading.classList.add('hidden');
        }
    });
}

// --- VERIFIER LOGIC ---
function setupVerifier() {
    const loading = document.getElementById('verifier-loading');
    const results = document.getElementById('verifier-results');

    setupDropzone('verifier-dropzone', 'verifier-file', async (file) => {
        results.classList.add('hidden');
        loading.classList.remove('hidden');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch(`${API_BASE_URL}/verify-hash`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('API Error');
            const data = await response.json();

            document.getElementById('res-verify-hash').innerText = data.sha256_hash;
            
            loading.classList.add('hidden');
            results.classList.remove('hidden');

        } catch (error) {
            console.error(error);
            alert("Verification failed.");
            loading.classList.add('hidden');
        }
    });
}

// --- SIMULATOR LOGIC ---
function setupSimulator() {
    const loading = document.getElementById('simulator-loading');
    const results = document.getElementById('simulator-results');

    setupDropzone('simulator-dropzone', 'simulator-file', async (file) => {
        results.classList.add('hidden');
        loading.classList.remove('hidden');
        
        // Reset progress bar
        document.getElementById('res-sim-confidence-bar').style.width = '0%';

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch(`${API_BASE_URL}/predict-future-attack`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('API Error');
            const data = await response.json();

            updateThreatBadge('res-sim-risk', data.future_attack_risk);
            document.getElementById('res-sim-type').innerText = data.predicted_attack_type;
            
            // Animate confidence bar
            setTimeout(() => {
                const confidencePct = data.confidence * 100;
                document.getElementById('res-sim-confidence-bar').style.width = `${confidencePct}%`;
                document.getElementById('res-sim-confidence-text').innerText = `${confidencePct.toFixed(1)}% Certainty`;
            }, 100);

            loading.classList.add('hidden');
            results.classList.remove('hidden');

        } catch (error) {
            console.error(error);
            alert("Simulation failed.");
            loading.classList.add('hidden');
        }
    });
}
