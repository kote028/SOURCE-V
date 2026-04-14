// =========================
// GLOBALS
// =========================
var radarChart = null;
let filesScannedCount = 0;
let threatsInterceptedCount = 0;
let verificationCount = 0;

const scanLog = [];
const threatLog = [];
const chainLog = [];

// =========================
// INIT
// =========================
document.addEventListener('DOMContentLoaded', () => {
    console.log("✅ JS LOADED");
    setupNavigation();
    setupScanner();
    setupVerifier();
    setupSimulator();
    setupWallet();
});

// =========================
// NAVIGATION
// =========================
function setupNavigation() {
    const navItems = document.querySelectorAll('.nav-links li');
    const sections = document.querySelectorAll('.page-section');

    const titleMap = {
        dashboard: { title: 'Dashboard Overview', sub: 'Real-time AI threat analysis and blockchain integrity' },
        scanner: { title: 'Deepfake Media Scanner', sub: 'Upload media for multi-modal AI ensemble analysis' },
        verifier: { title: 'Blockchain Integrity Verifier', sub: 'Tamper-proof cryptographic hash generation' },
        simulator: { title: 'Adversarial Threat Simulator', sub: 'FGSM/PGD predictive attack modeling engine' }
    };

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const target = item.dataset.target;
            navItems.forEach(i => i.classList.remove('active'));
            sections.forEach(s => s.classList.remove('active'));
            item.classList.add('active');
            const section = document.getElementById(target);
            if (section) section.classList.add('active');

            const info = titleMap[target];
            if (info) {
                document.getElementById('page-title').textContent = info.title;
                document.getElementById('page-subtitle').textContent = info.sub;
            }
        });
    });
}

// =========================
// SCANNER
// =========================
function setupScanner() {
    const loading = document.getElementById('scanner-loading');
    const results = document.getElementById('scanner-results');
    const dropzone = document.getElementById('scanner-dropzone');
    const fileInput = document.getElementById('scanner-file');

    if (!dropzone || !fileInput) {
        console.error("❌ Scanner dropzone or file input missing");
        return;
    }

    // Browse button
    const browseBtn = document.getElementById('scanner-browse-btn') || dropzone.querySelector("button");
    if (browseBtn) {
        browseBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            fileInput.click();
        });
    }

    // Drag & Drop
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add("hover");
    });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove("hover"));
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove("hover");
        const file = e.dataTransfer.files[0];
        if (file) handleScanFile(file);
    });

    // File input change
    fileInput.addEventListener('change', () => {
        const file = fileInput.files[0];
        if (file) handleScanFile(file);
        fileInput.value = ''; // reset so same file can be re-selected
    });

    async function handleScanFile(file) {
        console.log("📁 Scanner file:", file.name);
        results.classList.add('hidden');
        loading.classList.remove('hidden');
        animateLoadingSteps('loading-step');

        const formData = new FormData();
        formData.append('file', file); // backend expects 'file' field

        try {
            // Correct endpoint: /upload-media
            const response = await fetch('/upload-media', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status} ${response.statusText}`);
            }

            const data = await response.json();
            console.log("✅ API RESULT:", data);
            updateScanUI(data, file);

        } catch (err) {
            console.error("❌ ERROR:", err);
            alert(`Backend connection failed.\n\nMake sure the server is running:\n  python main.py\n\nError: ${err.message}`);
        }

        loading.classList.add('hidden');
        results.classList.remove('hidden');
    }
}

// =========================
// SCANNER UI UPDATE
// =========================
function updateScanUI(data, file) {
    // Map backend fields correctly
    // Backend returns: detection_verdict, fake_score, breakdown, sha256_hash, perceptual_hash, ipfs_cid, file_size_mb
    const verdict = data.detection_verdict || data.verdict || "UNKNOWN";
    const score = data.fake_score ?? data.final_score ?? 0;
    const breakdown = data.breakdown || {};
    const moduleScores = {
        gaze: breakdown.gaze ?? 0,
        lip_sync: breakdown.lip_sync ?? 0,
        voice: breakdown.voice ?? 0,
        emotion: breakdown.emotion ?? 0,
        behavioral: breakdown.behavioral ?? 0.5
    };

    const threat = verdict === "FAKE" ? "HIGH" : verdict === "REAL" ? "LOW" : "MEDIUM";

    // Verdict banner
    document.getElementById('res-scan-verdict').innerText = verdict;
    document.getElementById('res-scan-score').innerText = (score * 100).toFixed(1) + '%';
    document.getElementById('res-scan-threat').innerText = threat;

    // Threat badge color
    const threatBadge = document.getElementById('res-scan-threat');
    threatBadge.className = 'threat-badge';
    if (threat === 'HIGH') threatBadge.classList.add('high');
    else if (threat === 'MEDIUM') threatBadge.classList.add('medium');
    else threatBadge.classList.add('low');

    // Verdict banner color
    const verdictBanner = document.getElementById('verdict-banner');
    if (verdictBanner) {
        verdictBanner.className = 'verdict-banner';
        if (verdict === 'FAKE') verdictBanner.classList.add('fake');
        else if (verdict === 'REAL') verdictBanner.classList.add('real');
    }

    // Forensic ledger fields
    document.getElementById('res-scan-filename').innerText = file.name;
    document.getElementById('res-scan-size').innerText = data.file_size_mb?.toFixed(2) ?? (file.size / 1048576).toFixed(2);
    document.getElementById('res-scan-hash').innerText = data.sha256_hash || '-';
    document.getElementById('res-scan-phash').innerText = data.perceptual_hash || '-';
    document.getElementById('res-scan-ipfs').innerText = data.ipfs_cid || '-';

    // Module bars
    setModuleBar('bar-gaze', 'res-gaze', moduleScores.gaze);
    setModuleBar('bar-lipsync', 'res-lipsync', moduleScores.lip_sync);
    setModuleBar('bar-voice', 'res-voice', moduleScores.voice);
    setModuleBar('bar-emotion', 'res-emotion', moduleScores.emotion);
    setModuleBar('bar-behavioral', 'res-behavioral', moduleScores.behavioral);

    drawRadar(moduleScores);
    updateDashboard(threat, file.name, verdict, score, data);
}

// =========================
// VERIFIER
// =========================
function setupVerifier() {
    const loading = document.getElementById('verifier-loading');
    const results = document.getElementById('verifier-results');
    const dropzone = document.getElementById('verifier-dropzone');
    const fileInput = document.getElementById('verifier-file');

    if (!dropzone || !fileInput) {
        console.error("❌ Verifier dropzone or file input missing");
        return;
    }

    const btn = dropzone.querySelector('button');
    if (btn) {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            fileInput.click();
        });
    }

    dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('hover'); });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('hover'));
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('hover');
        const file = e.dataTransfer.files[0];
        if (file) handleVerifyFile(file);
    });

    fileInput.addEventListener('change', () => {
        const file = fileInput.files[0];
        if (file) handleVerifyFile(file);
        fileInput.value = '';
    });

    async function handleVerifyFile(file) {
        console.log("🔗 Verifier file:", file.name);
        results.classList.add('hidden');
        loading.classList.remove('hidden');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/verify-hash', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error(`Server error: ${response.status}`);

            const data = await response.json();
            console.log("✅ Verifier result:", data);

            document.getElementById('res-verify-hash').innerText = data.sha256_hash || '-';
            document.getElementById('res-verify-phash').innerText = data.perceptual_hash || '-';

            // Update blockchain verification count
            verificationCount++;
            document.getElementById('stat-verifications').innerText = verificationCount;
            const modalCount = document.getElementById('modal-chain-count');
            if (modalCount) modalCount.innerText = verificationCount;

            const now = new Date().toLocaleTimeString();
            chainLog.unshift({ name: file.name, time: now });
            renderLog('modal-chain-log', chainLog, 'green');

        } catch (err) {
            console.error("❌ Verifier error:", err);
            alert(`Verification failed.\n\nError: ${err.message}`);
        }

        loading.classList.add('hidden');
        results.classList.remove('hidden');
    }
}

// =========================
// SIMULATOR
// =========================
function setupSimulator() {
    const loading = document.getElementById('simulator-loading');
    const results = document.getElementById('simulator-results');
    const dropzone = document.getElementById('simulator-dropzone');
    const fileInput = document.getElementById('simulator-file');

    if (!dropzone || !fileInput) {
        console.error("❌ Simulator dropzone or file input missing");
        return;
    }

    const btn = dropzone.querySelector('button');
    if (btn) {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            fileInput.click();
        });
    }

    dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('hover'); });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('hover'));
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('hover');
        const file = e.dataTransfer.files[0];
        if (file) handleSimFile(file);
    });

    fileInput.addEventListener('change', () => {
        const file = fileInput.files[0];
        if (file) handleSimFile(file);
        fileInput.value = '';
    });

    async function handleSimFile(file) {
        console.log("⚡ Simulator file:", file.name);
        results.classList.add('hidden');
        loading.classList.remove('hidden');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/predict-future-attack', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error(`Server error: ${response.status}`);

            const data = await response.json();
            console.log("✅ Simulator result:", data);

            document.getElementById('res-sim-risk').innerText = data.future_attack_risk || '-';
            document.getElementById('res-sim-type').innerText = data.predicted_attack_type || '-';

            const confidence = (data.confidence || 0) * 100;
            const bar = document.getElementById('res-sim-confidence-bar');
            const txt = document.getElementById('res-sim-confidence-text');
            if (bar) bar.style.width = confidence.toFixed(1) + '%';
            if (txt) txt.innerText = confidence.toFixed(1) + '%';

            // Color risk badge
            const riskBadge = document.getElementById('res-sim-risk');
            if (riskBadge) {
                riskBadge.className = 'threat-badge';
                const r = data.future_attack_risk;
                if (r === 'HIGH') riskBadge.classList.add('high');
                else if (r === 'MEDIUM') riskBadge.classList.add('medium');
                else riskBadge.classList.add('low');
            }

        } catch (err) {
            console.error("❌ Simulator error:", err);
            alert(`Simulation failed.\n\nError: ${err.message}`);
        }

        loading.classList.add('hidden');
        results.classList.remove('hidden');
    }
}

// =========================
// WALLET (MetaMask)
// =========================
function setupWallet() {
    const btn = document.getElementById('connect-wallet-btn');
    if (!btn) return;

    btn.addEventListener('click', async () => {
        if (typeof window.ethereum === 'undefined') {
            alert('MetaMask is not installed. Please install it from metamask.io');
            return;
        }
        try {
            const accounts = await window.ethereum.request({ method: 'eth_requestAccounts' });
            const addr = accounts[0];
            const short = addr.slice(0, 6) + '...' + addr.slice(-4);

            document.getElementById('connect-wallet-btn').style.display = 'none';
            const connected = document.getElementById('wallet-connected');
            const addrEl = document.getElementById('wallet-address');
            const netEl = document.getElementById('network-info');
            if (connected) connected.style.display = 'flex';
            if (addrEl) addrEl.innerText = short;
            if (netEl) netEl.style.display = 'flex';

        } catch (err) {
            console.error("❌ MetaMask:", err);
        }
    });
}

// =========================
// HELPERS
// =========================
function setModuleBar(barId, pctId, score) {
    const pct = (score * 100).toFixed(1);
    const pctEl = document.getElementById(pctId);
    const barEl = document.getElementById(barId);
    if (pctEl) pctEl.innerText = pct + '%';
    if (barEl) barEl.style.width = pct + '%';
}

function updateDashboard(threat, fileName, verdict, score, data) {
    filesScannedCount++;
    if (threat !== 'LOW') threatsInterceptedCount++;

    document.getElementById('stat-files-scanned').innerText = filesScannedCount;
    document.getElementById('stat-threats-intercepted').innerText = threatsInterceptedCount;

    const modalScan = document.getElementById('modal-scan-count');
    const modalThreat = document.getElementById('modal-threat-count');
    if (modalScan) modalScan.innerText = filesScannedCount;
    if (modalThreat) modalThreat.innerText = threatsInterceptedCount;

    const now = new Date().toLocaleTimeString();
    const entry = { name: fileName, verdict, score: (score * 100).toFixed(1), time: now };
    scanLog.unshift(entry);
    if (threat !== 'LOW') threatLog.unshift(entry);

    renderLog('modal-scan-log', scanLog, verdict === 'FAKE' ? 'red' : 'green');
    renderLog('modal-threat-log', threatLog, 'red');

    // Also log blockchain
    verificationCount++;
    document.getElementById('stat-verifications').innerText = verificationCount;
    const chainCount = document.getElementById('modal-chain-count');
    if (chainCount) chainCount.innerText = verificationCount;

    if (data) {
        chainLog.unshift({ name: fileName, time: now, cid: data.ipfs_cid });
        renderLog('modal-chain-log', chainLog, 'green');
    }
}

function renderLog(containerId, log, color) {
    const el = document.getElementById(containerId);
    if (!el) return;
    if (log.length === 0) {
        el.innerHTML = '<div class="log-empty">No records yet.</div>';
        return;
    }
    el.innerHTML = log.slice(0, 10).map(e =>
        `<div class="log-entry" style="border-left: 3px solid var(--accent-${color === 'red' ? 'danger' : 'primary'});">
            <span class="log-name">${e.name}</span>
            <span class="log-meta">${e.verdict ? `${e.verdict} · ${e.score}%` : (e.cid ? e.cid.slice(0, 20) + '…' : '')} · ${e.time}</span>
        </div>`
    ).join('');
}

function animateLoadingSteps(elementId) {
    const steps = [
        'Initializing neural pipeline...',
        'Extracting visual frames...',
        'Running Gaze & Lip Sync analysis...',
        'Processing Voice & Emotion modules...',
        'Fusing ensemble scores...',
        'Generating blockchain forensics...'
    ];
    const el = document.getElementById(elementId);
    if (!el) return;
    let i = 0;
    const interval = setInterval(() => {
        if (i < steps.length) {
            el.textContent = steps[i++];
        } else {
            clearInterval(interval);
        }
    }, 600);
}

// =========================
// RADAR
// =========================
function drawRadar(breakdown) {
    const canvas = document.getElementById('radarChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    const scores = [
        breakdown.gaze,
        breakdown.lip_sync,
        breakdown.voice,
        breakdown.emotion,
        breakdown.behavioral
    ].map(v => (v || 0) * 100);

    if (radarChart) radarChart.destroy();

    radarChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Gaze', 'Lip Sync', 'Voice', 'Emotion', 'Behavior'],
            datasets: [{
                label: 'AI Analysis',
                data: scores,
                backgroundColor: 'rgba(99, 179, 237, 0.15)',
                borderColor: 'rgba(99, 179, 237, 0.9)',
                pointBackgroundColor: 'rgba(99, 179, 237, 1)',
                borderWidth: 2
            }]
        },
        options: {
            scales: {
                r: {
                    min: 0,
                    max: 100,
                    ticks: { color: '#94a3b8', stepSize: 25, backdropColor: 'transparent' },
                    grid: { color: 'rgba(148,163,184,0.15)' },
                    pointLabels: { color: '#e2e8f0', font: { size: 12 } }
                }
            },
            plugins: {
                legend: { labels: { color: '#e2e8f0' } }
            }
        }
    });
}