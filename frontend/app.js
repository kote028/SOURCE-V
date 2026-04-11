const API_BASE_URL = 'http://127.0.0.1:5000';
let radarChart = null;
let filesScannedCount = 0;
let threatsInterceptedCount = 0;
let verificationsCount = 0;
const scanLog = [];       // { name, verdict, threat }
const threatLog = [];     // { name, threat, score }
const chainLog = [];      // { name, hash }

// ---- MODALS ----
function openModal(id) {
    // Sync live counts
    document.getElementById('modal-scan-count').innerText = filesScannedCount;
    document.getElementById('modal-threat-count').innerText = threatsInterceptedCount;
    document.getElementById('modal-chain-count').innerText = verificationsCount;

    // Render logs
    renderLog('modal-scan-log', scanLog, (e) =>
        `<span><i class="fa-solid fa-file" style="color:var(--accent-primary);margin-right:6px"></i>${e.name}</span><span class="threat-badge ${e.threat}" style="font-size:0.75rem;padding:3px 8px">${e.verdict}</span>`
    );
    renderLog('modal-threat-log', threatLog, (e) =>
        `<span><i class="fa-solid fa-triangle-exclamation" style="color:var(--danger);margin-right:6px"></i>${e.name}</span><span class="threat-badge ${e.threat}" style="font-size:0.75rem;padding:3px 8px">${e.threat}</span>`
    );
    renderLog('modal-chain-log', chainLog, (e) =>
        `<span><i class="fa-solid fa-lock" style="color:var(--accent-secondary);margin-right:6px"></i>${e.name}</span><span style="font-family:monospace;font-size:0.72rem;color:var(--text-secondary)">${e.hash.substring(0,16)}...</span>`
    );

    document.getElementById(id).classList.add('open');
}

function closeModal(id) {
    document.getElementById(id).classList.remove('open');
}

function renderLog(containerId, data, rowFn) {
    const el = document.getElementById(containerId);
    if (!el) return;
    if (data.length === 0) {
        el.innerHTML = '<div class="modal-log-empty">No records yet this session.</div>';
    } else {
        el.innerHTML = [...data].reverse().map(e =>
            `<div class="modal-log-entry">${rowFn(e)}</div>`
        ).join('');
    }
}

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        ['modal-scanned','modal-threats','modal-chain'].forEach(id => closeModal(id));
    }
});

document.addEventListener('DOMContentLoaded', () => {
    setupNavigation();
    setupWallet();
    setupScanner();
    setupVerifier();
    setupSimulator();
    initParticles();
});

// ---- PARTICLES ----
function initParticles() {
    const canvas = document.getElementById('particles-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    window.addEventListener('resize', () => { canvas.width = window.innerWidth; canvas.height = window.innerHeight; });

    const particles = Array.from({ length: 60 }, () => ({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.4,
        vy: (Math.random() - 0.5) * 0.4,
        r: Math.random() * 1.5 + 0.5,
        alpha: Math.random() * 0.5 + 0.1,
    }));

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        particles.forEach(p => {
            p.x += p.vx; p.y += p.vy;
            if (p.x < 0) p.x = canvas.width;
            if (p.x > canvas.width) p.x = 0;
            if (p.y < 0) p.y = canvas.height;
            if (p.y > canvas.height) p.y = 0;
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(139, 92, 246, ${p.alpha})`;
            ctx.fill();
        });
        // Draw lines between nearby particles
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 120) {
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = `rgba(139, 92, 246, ${0.1 * (1 - dist / 120)})`;
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            }
        }
        requestAnimationFrame(draw);
    }
    draw();
}

// ---- NAVIGATION ----
function setupNavigation() {
    const navLinks = document.querySelectorAll('.nav-links li');
    const sections = document.querySelectorAll('.page-section');
    const pageTitle = document.getElementById('page-title');
    const pageSub = document.getElementById('page-subtitle');

    const meta = {
        'dashboard': ['Dashboard Overview', 'Real-time AI threat analysis and blockchain integrity'],
        'scanner': ['Deepfake Media Scanner', 'Multi-modal AI ensemble — 5 detection modules + score fusion'],
        'verifier': ['Blockchain Integrity Verifier', 'Tamper-proof cryptographic hash & IPFS forensic ledger'],
        'simulator': ['Adversarial Threat Simulator', 'FGSM/PGD proactive threat modeling & attack prediction'],
    };

    navLinks.forEach(link => {
        link.addEventListener('click', () => {
            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');
            const id = link.getAttribute('data-target');
            sections.forEach(s => s.classList.remove('active'));
            document.getElementById(id).classList.add('active');
            if (meta[id]) { pageTitle.innerText = meta[id][0]; pageSub.innerText = meta[id][1]; }
        });
    });
}

// ---- WALLET ----
function setupWallet() {
    const btn = document.getElementById('connect-wallet-btn');
    const walletDiv = document.getElementById('wallet-connected');
    const addrSpan = document.getElementById('wallet-address');
    const netInfo = document.getElementById('network-info');
    const netName = document.getElementById('network-name');

    if (!btn) return;

    btn.addEventListener('click', async () => {
        if (typeof window.ethereum !== 'undefined') {
            try {
                const accounts = await window.ethereum.request({ method: 'eth_requestAccounts' });
                const account = accounts[0];
                
                // Get network info
                const chainId = await window.ethereum.request({ method: 'eth_chainId' });
                const networkMap = { '0x539': 'Ganache Local', '0x1': 'Ethereum Mainnet', '0x5': 'Goerli Testnet', '0xaa36a7': 'Sepolia' };
                
                btn.style.display = 'none';
                walletDiv.style.display = 'flex';
                addrSpan.innerText = `${account.substring(0, 6)}...${account.substring(account.length - 4)}`;

                netInfo.style.display = 'flex';
                netName.innerText = networkMap[chainId] || `Chain ${parseInt(chainId, 16)}`;

            } catch (err) {
                console.error('Wallet error:', err);
            }
        } else {
            alert('MetaMask not detected! Please install MetaMask and connect it to your Ganache network first.');
        }
    });
}

// ---- UTILITIES ----
function setupDropzone(dropzoneId, fileInputId, callback) {
    const zone = document.getElementById(dropzoneId);
    const input = document.getElementById(fileInputId);

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(e => zone.addEventListener(e, ev => { ev.preventDefault(); ev.stopPropagation(); }));
    ['dragenter', 'dragover'].forEach(e => zone.addEventListener(e, () => zone.classList.add('dragover')));
    ['dragleave', 'drop'].forEach(e => zone.addEventListener(e, () => zone.classList.remove('dragover')));
    zone.addEventListener('drop', e => { const f = e.dataTransfer.files[0]; if (f) callback(f); });
    input.addEventListener('change', function () { if (this.files[0]) callback(this.files[0]); });
}

function setModuleBar(barId, pctId, score) {
    const pct = (score * 100).toFixed(1);
    const bar = document.getElementById(barId);
    const el = document.getElementById(pctId);
    el.innerText = pct + '%';
    setTimeout(() => {
        bar.style.width = pct + '%';
        bar.className = 'module-bar' + (score > 0.6 ? ' danger' : '');
    }, 100);
}

function updateDashboard(threatLevel, fileName, verdict, sha256) {
    filesScannedCount++;
    document.getElementById('stat-files-scanned').innerText = filesScannedCount;
    scanLog.push({ name: fileName, verdict, threat: threatLevel });

    if (threatLevel === 'HIGH' || threatLevel === 'MEDIUM') {
        threatsInterceptedCount++;
        document.getElementById('stat-threats-intercepted').innerText = threatsInterceptedCount;
        threatLog.push({ name: fileName, threat: threatLevel });
    }
    verificationsCount++;
    document.getElementById('stat-verifications').innerText = verificationsCount;
    if (sha256) chainLog.push({ name: fileName, hash: sha256 });
}

// ---- RADAR CHART ----
function drawRadar(breakdown) {
    const ctx = document.getElementById('radarChart').getContext('2d');
    const data = [breakdown.gaze, breakdown.lip_sync, breakdown.voice, breakdown.emotion, breakdown.behavioral].map(v => v * 100);

    if (radarChart) radarChart.destroy();
    radarChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Gaze', 'Lip Sync', 'Voice', 'Emotion', 'Behavioral'],
            datasets: [{
                label: 'Fake Confidence %',
                data,
                backgroundColor: 'rgba(139, 92, 246, 0.2)',
                borderColor: 'rgba(139, 92, 246, 0.9)',
                pointBackgroundColor: '#06b6d4',
                pointBorderColor: '#06b6d4',
                pointRadius: 5,
                borderWidth: 2,
            }]
        },
        options: {
            responsive: true,
            scales: {
                r: {
                    min: 0, max: 100,
                    ticks: { color: '#64748b', stepSize: 25, font: { size: 10 } },
                    grid: { color: 'rgba(255,255,255,0.06)' },
                    pointLabels: { color: '#94a3b8', font: { size: 12, family: 'Inter' } },
                    angleLines: { color: 'rgba(255,255,255,0.06)' },
                }
            },
            plugins: { legend: { display: false } }
        }
    });
}

// Loading step cycling
const loadingSteps = [
    'Initializing neural pipeline...',
    'Running OpenCV preprocessing...',
    'Analyzing gaze vectors (CNN/LSTM)...',
    'Detecting lip-sync anomalies...',
    'Processing voice MFCC spectrogram...',
    'Evaluating emotion action units...',
    'Behavioral temporal analysis...',
    'Running score fusion ensemble...',
    'Generating blockchain proof...',
];

function cycleLoadingText(elId) {
    let i = 0;
    const el = document.getElementById(elId);
    return setInterval(() => {
        if (el) el.innerText = loadingSteps[i % loadingSteps.length];
        i++;
    }, 900);
}

// ---- SCANNER ----
function setupScanner() {
    const loading = document.getElementById('scanner-loading');
    const results = document.getElementById('scanner-results');
    const dropzone = document.getElementById('scanner-dropzone');

    setupDropzone('scanner-dropzone', 'scanner-file', async (file) => {
        results.classList.add('hidden');
        loading.classList.remove('hidden');
        dropzone.classList.add('scanning');
        const cycleTimer = cycleLoadingText('loading-step');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch(`${API_BASE_URL}/upload-media`, { method: 'POST', body: formData });
            clearInterval(cycleTimer);
            if (!response.ok) throw new Error('API Error');
            const data = await response.json();

            // Verdict banner
            const banner = document.getElementById('verdict-banner');
            const verdictEl = document.getElementById('res-scan-verdict');
            const verdictIcon = document.getElementById('verdict-icon');

            banner.className = 'verdict-banner ' + (data.detection_verdict === 'FAKE' ? 'fake' : 'real');
            verdictEl.className = 'verdict-value ' + (data.detection_verdict === 'FAKE' ? 'fake' : 'real');
            verdictEl.innerText = data.detection_verdict;
            verdictIcon.innerHTML = data.detection_verdict === 'FAKE'
                ? '<i class="fa-solid fa-circle-xmark" style="color:#ef4444;font-size:2.5rem;filter:drop-shadow(0 0 12px rgba(239,68,68,0.6))"></i>'
                : '<i class="fa-solid fa-circle-check" style="color:#10b981;font-size:2.5rem;filter:drop-shadow(0 0 12px rgba(16,185,129,0.6))"></i>';

            // Score
            document.getElementById('res-scan-score').innerText = (data.fake_score * 100).toFixed(1) + '%';

            // Threat badge
            const tb = document.getElementById('res-scan-threat');
            tb.innerText = data.threat_prediction;
            tb.className = 'threat-badge ' + data.threat_prediction;

            // Module breakdown bars
            setModuleBar('bar-gaze', 'res-gaze', data.breakdown.gaze);
            setModuleBar('bar-lipsync', 'res-lipsync', data.breakdown.lip_sync);
            setModuleBar('bar-voice', 'res-voice', data.breakdown.voice);
            setModuleBar('bar-emotion', 'res-emotion', data.breakdown.emotion);
            setModuleBar('bar-behavioral', 'res-behavioral', data.breakdown.behavioral);

            // Radar chart
            drawRadar(data.breakdown);

            // Hashes
            document.getElementById('res-scan-filename').innerText = data.file_name;
            document.getElementById('res-scan-size').innerText = data.file_size_mb;
            document.getElementById('res-scan-hash').innerText = data.sha256_hash;
            document.getElementById('res-scan-phash').innerText = data.perceptual_hash || 'N/A';
            document.getElementById('res-scan-ipfs').innerText = data.ipfs_cid || 'N/A';

            updateDashboard(data.threat_prediction);

            dropzone.classList.remove('scanning');
            loading.classList.add('hidden');
            results.classList.remove('hidden');

        } catch (err) {
            clearInterval(cycleTimer);
            dropzone.classList.remove('scanning');
            loading.classList.add('hidden');
            console.error(err);
            alert('Analysis failed. Make sure the backend is running on http://127.0.0.1:5000');
        }
    });
}

// ---- VERIFIER ----
function setupVerifier() {
    const loading = document.getElementById('verifier-loading');
    const results = document.getElementById('verifier-results');

    setupDropzone('verifier-dropzone', 'verifier-file', async (file) => {
        results.classList.add('hidden');
        loading.classList.remove('hidden');
        const formData = new FormData();
        formData.append('file', file);
        try {
            const response = await fetch(`${API_BASE_URL}/verify-hash`, { method: 'POST', body: formData });
            if (!response.ok) throw new Error('API Error');
            const data = await response.json();
            document.getElementById('res-verify-hash').innerText = data.sha256_hash;
            document.getElementById('res-verify-phash').innerText = data.perceptual_hash || 'N/A';
            verificationsCount++;
            document.getElementById('stat-verifications').innerText = verificationsCount;
            loading.classList.add('hidden');
            results.classList.remove('hidden');
        } catch (err) {
            loading.classList.add('hidden');
            console.error(err);
            alert('Verification failed.');
        }
    });
}

// ---- SIMULATOR ----
function setupSimulator() {
    const loading = document.getElementById('simulator-loading');
    const results = document.getElementById('simulator-results');

    setupDropzone('simulator-dropzone', 'simulator-file', async (file) => {
        results.classList.add('hidden');
        loading.classList.remove('hidden');
        document.getElementById('res-sim-confidence-bar').style.width = '0%';
        const formData = new FormData();
        formData.append('file', file);
        try {
            const response = await fetch(`${API_BASE_URL}/predict-future-attack`, { method: 'POST', body: formData });
            if (!response.ok) throw new Error('API Error');
            const data = await response.json();

            const rb = document.getElementById('res-sim-risk');
            rb.innerText = data.future_attack_risk;
            rb.className = 'threat-badge ' + data.future_attack_risk;
            document.getElementById('res-sim-type').innerText = data.predicted_attack_type;

            setTimeout(() => {
                const pct = (data.confidence * 100).toFixed(1);
                document.getElementById('res-sim-confidence-bar').style.width = pct + '%';
                document.getElementById('res-sim-confidence-text').innerText = pct + '% prediction certainty';
            }, 150);

            loading.classList.add('hidden');
            results.classList.remove('hidden');
        } catch (err) {
            loading.classList.add('hidden');
            console.error(err);
            alert('Simulation failed.');
        }
    });
}
