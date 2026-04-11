// =========================
// CONFIG
// =========================
const API_BASE_URL = ""; // SAME SERVER

let radarChart = null;
let filesScannedCount = 0;
let threatsInterceptedCount = 0;

// =========================
// INIT
// =========================
document.addEventListener('DOMContentLoaded', () => {
    setupScanner();
});

// =========================
// SCANNER
// =========================
function setupScanner() {
    const loading = document.getElementById('scanner-loading');
    const results = document.getElementById('scanner-results');
    const dropzone = document.getElementById('scanner-dropzone');
    const fileInput = document.getElementById('scanner-file');

    if (!dropzone || !fileInput) {
        console.error("❌ Dropzone or file input missing");
        return;
    }

    // CLICK → open file picker
    dropzone.addEventListener('click', () => {
        fileInput.click();
    });

    // DRAG OVER
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add("hover");
    });

    // DRAG LEAVE
    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove("hover");
    });

    // DROP
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove("hover");

        const file = e.dataTransfer.files[0];
        if (file) handleFile(file);
    });

    // FILE SELECT
    fileInput.addEventListener('change', () => {
        const file = fileInput.files[0];
        if (file) handleFile(file);
    });

    // MAIN HANDLER
    async function handleFile(file) {
        console.log("📁 File selected:", file);

        results.classList.add('hidden');
        loading.classList.remove('hidden');

        const formData = new FormData();
        formData.append('video', file);

        try {
            const response = await fetch(`/analyze`, {
                method: 'POST',
                body: formData
            });

            console.log("Status:", response.status);

            const data = await response.json();
            console.log("API RESULT:", data);

            updateUI(data, file);

        } catch (err) {
            console.error("❌ ERROR:", err);
            alert("Backend connection failed");
        }

        loading.classList.add('hidden');
        results.classList.remove('hidden');
    }
}

// =========================
// UI UPDATE
// =========================
function updateUI(data, file) {

    const breakdown = {
        gaze: data.module_scores?.gaze || 0,
        lip_sync: data.module_scores?.lip_sync || 0,
        voice: data.module_scores?.voice || 0,
        emotion: data.module_scores?.emotion || 0,
        behavioral: 0.5
    };

    const verdict = data.verdict || "UNKNOWN";
    const score = data.final_score || 0;

    const threat =
        verdict === "FAKE" ? "HIGH" :
        verdict === "REAL" ? "LOW" : "MEDIUM";

    // TEXT
    document.getElementById('res-scan-verdict').innerText = verdict;
    document.getElementById('res-scan-score').innerText =
        (score * 100).toFixed(1) + '%';

    document.getElementById('res-scan-threat').innerText = threat;
    document.getElementById('res-scan-filename').innerText = file.name;

    // MODULE BARS
    setModuleBar('bar-gaze', 'res-gaze', breakdown.gaze);
    setModuleBar('bar-lipsync', 'res-lipsync', breakdown.lip_sync);
    setModuleBar('bar-voice', 'res-voice', breakdown.voice);
    setModuleBar('bar-emotion', 'res-emotion', breakdown.emotion);
    setModuleBar('bar-behavioral', 'res-behavioral', breakdown.behavioral);

    drawRadar(breakdown);

    updateDashboard(threat, file.name, verdict);
}

// =========================
// HELPERS
// =========================
function setModuleBar(barId, pctId, score) {
    const pct = (score * 100).toFixed(1);
    document.getElementById(pctId).innerText = pct + '%';
    document.getElementById(barId).style.width = pct + '%';
}

function updateDashboard(threat, name, verdict) {
    filesScannedCount++;

    if (threat !== 'LOW') {
        threatsInterceptedCount++;
    }

    document.getElementById('stat-files-scanned').innerText = filesScannedCount;
    document.getElementById('stat-threats-intercepted').innerText = threatsInterceptedCount;
}

// =========================
// RADAR CHART
// =========================
function drawRadar(breakdown) {
    const ctx = document.getElementById('radarChart').getContext('2d');

    const data = [
        breakdown.gaze,
        breakdown.lip_sync,
        breakdown.voice,
        breakdown.emotion,
        breakdown.behavioral
    ].map(v => v * 100);

    if (radarChart) radarChart.destroy();

    radarChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Gaze', 'Lip', 'Voice', 'Emotion', 'Behavior'],
            datasets: [{
                label: 'AI Analysis',
                data: data
            }]
        }
    });
}