/**
 * ═══════════════════════════════════════════════════════════
 *  SVVAP — Spatial Visual Verification & Analysis Platform
 *  app.js — State management, routing, API, 3D visualizer
 * ═══════════════════════════════════════════════════════════
 */

'use strict';

// ─── CONFIG ──────────────────────────────────────────────────────────────────

const API_ENDPOINT  = 'http://localhost:8000/analyze';
const USE_MOCK_DATA = false; // Set true to always use mock regardless of file
const MOCK_RESPONSE = {
  status: 'success',
  analysis: {
    is_deepfake:    true,
    logical_gaze:   false,
    distance:       1.6635,
    vergence_point: [0.5, -1.2, 8.4]
  },
  raw_coordinates: {
    left_pupil:  [-1,  0, 0],
    right_pupil: [ 1,  0, 0],
    left_gaze:   [ 1,  1, 10],
    right_gaze:  [-1, -2, 10]
  },
  av_sync_data: { score: 0.42 }   // <-- ADD THIS LINE
};
// ─── STATE ───────────────────────────────────────────────────────────────────

const state = {
  file:          null,
  result:        null,
  analysisCount: 0,
  fakeCount:     0,
  phase:         'idle'   // idle | loading | results
};

// ─── DOM ─────────────────────────────────────────────────────────────────────

const $ = id => document.getElementById(id);
const statAvSync   = $('statAvSync');

// Pages
const landingPage    = $('landingPage');
const dashboardPage  = $('dashboardPage');
const landingCTA     = $('landingCTA');
const btnBack        = $('btnBack');

// Input
const dropZone         = $('dropZone');
const fileInput        = $('fileInput');
const videoPlayer      = $('videoPlayer');
const videoPreviewWrap = $('videoPreviewWrap');
const videoScanOverlay = $('videoScanOverlay');
const btnAnalyze       = $('btnAnalyze');
const btnDemo          = $('btnDemo');

// Results
const resultsIdle    = $('resultsIdle');
const resultsLoading = $('resultsLoading');
const resultsContent = $('resultsContent');

// Verdict
const verdictBadge = $('verdictBadge');
const verdictText  = $('verdictText');
const verdictSub   = $('verdictSub');
const verdictIcon  = $('verdictIcon');

// Stats
const statDistance = $('statDistance');
const statLogical  = $('statLogical');
const statVergZ    = $('statVergZ');
const coordsGrid   = $('coordsGrid');

// Header
const analysisCountEl = $('analysisCount');
const detectRateEl    = $('detectRate');

// File meta
const metaName   = $('meta-name');
const metaSize   = $('meta-size');
const metaType   = $('meta-type');
const metaStatus = $('meta-status');
const vpFilename = $('vp-filename');
const vpSize     = $('vp-size');

// Visualizer
const btnExpand    = $('btnExpand');
const modalOverlay = $('modalOverlay');
const modalClose   = $('modalClose');

// ─── UTILITIES ───────────────────────────────────────────────────────────────

function formatBytes(bytes) {
  if (bytes < 1024)     return bytes + ' B';
  if (bytes < 1048576)  return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1048576).toFixed(1) + ' MB';
}

function setMetaStatus(label, cls) {
  metaStatus.textContent = label;
  metaStatus.className = 'meta-val meta-status' + (cls ? ' ' + cls : '');
}

function updateHeaderStats() {
  analysisCountEl.textContent = state.analysisCount;
  if (state.analysisCount === 0) {
    detectRateEl.textContent = '—';
  } else {
    const pct = Math.round((state.fakeCount / state.analysisCount) * 100);
    detectRateEl.textContent = pct + '%';
  }
}

// ─── PAGE ROUTING ────────────────────────────────────────────────────────────

function showDashboard() {
  landingPage.classList.add('exit');
  setTimeout(() => {
    landingPage.style.display = 'none';
    dashboardPage.style.display = 'flex';
    dashboardPage.classList.add('enter');
    // Remove animation class after it completes
    setTimeout(() => dashboardPage.classList.remove('enter'), 600);
  }, 500);
}

function showLanding() {
  dashboardPage.style.display = 'none';
  landingPage.style.display = 'flex';
  landingPage.classList.remove('exit');
  // Force reflow so animation re-triggers
  void landingPage.offsetWidth;
}

landingCTA.addEventListener('click', showDashboard);
btnBack.addEventListener('click', showLanding);

// ─── FILE HANDLING ───────────────────────────────────────────────────────────

function handleFile(file) {
  if (!file || !file.type.startsWith('video/')) {
    alert('Please upload a valid video file (MP4, MOV, AVI, WEBM).');
    return;
  }

  state.file = file;

  // Update meta
  metaName.textContent = file.name;
  metaSize.textContent = formatBytes(file.size);
  metaType.textContent = file.type || 'video/unknown';
  setMetaStatus('READY TO ANALYZE', 'ready');

  // Preview
  const url = URL.createObjectURL(file);
  videoPlayer.src = url;
  vpFilename.textContent = file.name;
  vpSize.textContent = formatBytes(file.size);
  videoPreviewWrap.classList.add('visible');

  // Enable button & reset results
  btnAnalyze.disabled = false;
  showPhase('idle');
}

// ─── DROP ZONE EVENTS ────────────────────────────────────────────────────────

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') fileInput.click(); });
fileInput.addEventListener('change', e => { if (e.target.files[0]) handleFile(e.target.files[0]); });

dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

// ─── PHASE MANAGEMENT ────────────────────────────────────────────────────────

function showPhase(phase) {
  state.phase = phase;
  resultsIdle.style.display    = phase === 'idle'    ? 'flex' : 'none';
  resultsLoading.style.display = phase === 'loading' ? 'flex' : 'none';
  resultsContent.style.display = phase === 'results' ? 'flex' : 'none';
}

// ─── LOADING STEP SEQUENCER ──────────────────────────────────────────────────

const STEP_IDS = ['lstep1', 'lstep2', 'lstep3', 'lstep4', 'lstep5'];
let stepTimers = [];

function runLoadingSteps() {
  clearLoadingSteps();

  STEP_IDS.forEach(id => {
    const el = $(id);
    el.classList.remove('active', 'done');
  });

  STEP_IDS.forEach((id, i) => {
    const t1 = setTimeout(() => {
      if (i > 0) {
        const prev = $(STEP_IDS[i - 1]);
        prev.classList.remove('active');
        prev.classList.add('done');
      }
      $(id).classList.add('active');
    }, i * 500);
    stepTimers.push(t1);
  });
}

function clearLoadingSteps() {
  stepTimers.forEach(clearTimeout);
  stepTimers = [];
  STEP_IDS.forEach(id => {
    const el = $(id);
    el.classList.remove('active', 'done');
  });
}

// ─── API CALL ────────────────────────────────────────────────────────────────

async function analyzeVideo() {
  if (!state.file && !USE_MOCK_DATA) return;

  showPhase('loading');
  btnAnalyze.disabled = true;
  btnDemo.disabled = true;
  setMetaStatus('ANALYZING...', 'analyzing');
  videoScanOverlay.classList.add('scanning');
  runLoadingSteps();

  try {
    let data;

    if (USE_MOCK_DATA || !state.file) {
      // Simulate realistic API delay for the demo
      await new Promise(r => setTimeout(r, 3000));
      data = MOCK_RESPONSE;
    } else {
      const formData = new FormData();
      formData.append('file', state.file);

      const response = await fetch(API_ENDPOINT, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}: ${response.statusText}`);
      }

      data = await response.json();
    }

    if (data.status !== 'success') {
      throw new Error(data.message || 'Analysis failed — unknown server error.');
    }

    clearLoadingSteps();
    videoScanOverlay.classList.remove('scanning');

    // Update session state
    state.result = data;
    state.analysisCount++;
    if (data.analysis.is_deepfake) state.fakeCount++;
    updateHeaderStats();

    setMetaStatus(
      data.analysis.is_deepfake ? 'DEEPFAKE DETECTED' : 'AUTHENTIC',
      data.analysis.is_deepfake ? 'done-fake' : 'done-real'
    );

    renderResults(data);
    showPhase('results');

  } catch (err) {
    clearLoadingSteps();
    videoScanOverlay.classList.remove('scanning');
    showPhase('idle');
    setMetaStatus('ERROR — RETRY', '');
    console.error('[SVVAP] Error:', err);
    alert(`Analysis failed:\n${err.message}\n\nTip: Use the ⚡ DEMO button to test the UI with mock data.`);
  }

  btnAnalyze.disabled = !state.file;
  btnDemo.disabled = false;
}

// ─── RENDER RESULTS ──────────────────────────────────────────────────────────

function renderResults(data) {
  const { analysis, raw_coordinates } = data;
  const isDeepfake = analysis.is_deepfake;

  // Verdict badge
  verdictBadge.className = 'verdict-badge ' + (isDeepfake ? 'fake' : 'real');
  verdictIcon.textContent = isDeepfake ? '⚠' : '✓';
  verdictText.textContent = isDeepfake ? 'DEEPFAKE DETECTED' : 'AUTHENTIC';
  verdictSub.textContent  = isDeepfake
    ? `Gaze anomaly distance: ${analysis.distance.toFixed(4)} — exceeds 0.500 threshold`
    : `Gaze anomaly distance: ${analysis.distance.toFixed(4)} — within 0.500 threshold`;

  // Animate badge in
  verdictBadge.classList.remove('revealed');
  void verdictBadge.offsetWidth;
  verdictBadge.classList.add('revealed');

  // Stats
  statDistance.textContent = analysis.distance.toFixed(4);
  statDistance.style.color = isDeepfake ? 'var(--red)' : 'var(--cyan)';

  statLogical.textContent = analysis.logical_gaze ? 'TRUE' : 'FALSE';
  statLogical.style.color = analysis.logical_gaze ? 'var(--cyan)' : 'var(--red)';

  const vp = analysis.vergence_point;
  statVergZ.textContent = vp ? vp[2].toFixed(3) : '∞';
  // AV Sync score
  if (data.av_sync_data && statAvSync) {
    const score = data.av_sync_data.score;
    statAvSync.textContent = score.toFixed(3);
    statAvSync.style.color = score >= 0.7 ? 'var(--cyan)' : 'var(--red)';
  }
  // Coordinates panel
  renderCoordinates(raw_coordinates, analysis.vergence_point);

  // 3D Visualizer
  renderGazeVisualizer('gazeVisualizer', raw_coordinates, analysis.vergence_point, isDeepfake);
}

// ─── COORDINATES PANEL ───────────────────────────────────────────────────────

function renderCoordinates(coords, vergencePoint) {
  const entries = [
    { label: 'LEFT PUPIL',     val: coords.left_pupil,  color: '#4f8ef7' },
    { label: 'RIGHT PUPIL',    val: coords.right_pupil, color: '#ff8c42' },
    { label: 'LEFT GAZE DIR',  val: coords.left_gaze,   color: '#4f8ef7' },
    { label: 'RIGHT GAZE DIR', val: coords.right_gaze,  color: '#ff8c42' },
    { label: 'VERGENCE POINT', val: vergencePoint,       color: '#ff3356' },
  ];

  coordsGrid.innerHTML = entries.map(({ label, val, color }) => {
    const fmt = v => v ? `[${v.map(n => n.toFixed(2)).join(', ')}]` : '—';
    return `
      <div class="coord-cell">
        <span class="coord-label" style="color:${color}">${label}</span>
        <span class="coord-value">${fmt(val)}</span>
      </div>`;
  }).join('');
}

// ─── 3D GAZE VISUALIZER ──────────────────────────────────────────────────────

function buildGazeTraces(coords, vergencePoint) {
  const { left_pupil, right_pupil, left_gaze, right_gaze } = coords;

  // Gaze endpoints = pupil + direction vector (scale=1 matches raw data scale)
  const SCALE   = 1;
  const leftEnd  = left_pupil.map((v, i)  => v + left_gaze[i]  * SCALE);
  const rightEnd = right_pupil.map((v, i) => v + right_gaze[i] * SCALE);

  const traces = [];

  // ── Left gaze ray (Blue)
  traces.push({
    type: 'scatter3d', mode: 'lines',
    name: 'Left Gaze Ray',
    x: [left_pupil[0], leftEnd[0]],
    y: [left_pupil[1], leftEnd[1]],
    z: [left_pupil[2], leftEnd[2]],
    line: { color: '#4f8ef7', width: 7 },
    hovertemplate: 'Left Gaze Ray<br>x:%{x:.3f} y:%{y:.3f} z:%{z:.3f}<extra></extra>'
  });

  // ── Right gaze ray (Orange)
  traces.push({
    type: 'scatter3d', mode: 'lines',
    name: 'Right Gaze Ray',
    x: [right_pupil[0], rightEnd[0]],
    y: [right_pupil[1], rightEnd[1]],
    z: [right_pupil[2], rightEnd[2]],
    line: { color: '#ff8c42', width: 7 },
    hovertemplate: 'Right Gaze Ray<br>x:%{x:.3f} y:%{y:.3f} z:%{z:.3f}<extra></extra>'
  });

  // ── Left pupil origin
  traces.push({
    type: 'scatter3d', mode: 'markers',
    name: 'Left Pupil',
    x: [left_pupil[0]], y: [left_pupil[1]], z: [left_pupil[2]],
    marker: {
      size: 11, color: '#4f8ef7', symbol: 'circle',
      line: { color: '#93c5fd', width: 2 }
    },
    hovertemplate: 'Left Pupil<br>[%{x:.2f}, %{y:.2f}, %{z:.2f}]<extra></extra>'
  });

  // ── Right pupil origin
  traces.push({
    type: 'scatter3d', mode: 'markers',
    name: 'Right Pupil',
    x: [right_pupil[0]], y: [right_pupil[1]], z: [right_pupil[2]],
    marker: {
      size: 11, color: '#ff8c42', symbol: 'circle',
      line: { color: '#fdba74', width: 2 }
    },
    hovertemplate: 'Right Pupil<br>[%{x:.2f}, %{y:.2f}, %{z:.2f}]<extra></extra>'
  });

  // ── Ray arrowhead dots (endpoint markers)
  traces.push({
    type: 'scatter3d', mode: 'markers',
    name: 'Ray Endpoints',
    x: [leftEnd[0], rightEnd[0]],
    y: [leftEnd[1], rightEnd[1]],
    z: [leftEnd[2], rightEnd[2]],
    marker: {
      size: 5,
      color: ['#93c5fd', '#fdba74'],
      symbol: 'circle'
    },
    showlegend: false,
    hoverinfo: 'skip'
  });

  // ── Vergence point (Red Diamond) — most prominent marker
  if (vergencePoint) {
    traces.push({
      type: 'scatter3d', mode: 'markers+text',
      name: 'Vergence Point',
      x: [vergencePoint[0]], y: [vergencePoint[1]], z: [vergencePoint[2]],
      marker: {
        size: 16, color: '#ff3356', symbol: 'diamond',
        line: { color: '#ff8099', width: 2 }
      },
      text: ['VERGENCE'],
      textposition: 'top center',
      textfont: { color: '#ff3356', size: 9, family: 'Share Tech Mono' },
      hovertemplate: '◆ Vergence Point<br>[%{x:.3f}, %{y:.3f}, %{z:.3f}]<extra></extra>'
    });
  }

  // ── Divergence gap (dotted line between endpoints)
  traces.push({
    type: 'scatter3d', mode: 'lines',
    name: 'Divergence Gap',
    x: [leftEnd[0], rightEnd[0]],
    y: [leftEnd[1], rightEnd[1]],
    z: [leftEnd[2], rightEnd[2]],
    line: { color: 'rgba(255,255,255,0.15)', width: 2, dash: 'dot' },
    hoverinfo: 'skip', showlegend: false
  });

  return traces;
}

function getPlotLayout(subtitleText) {
  return {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor:  'rgba(0,0,0,0)',
    margin: { l: 0, r: 0, t: 28, b: 0 },
    annotations: subtitleText ? [{
      text: subtitleText,
      x: 0.01, y: 1.0, xref: 'paper', yref: 'paper',
      xanchor: 'left', yanchor: 'top',
      showarrow: false,
      font: { family: 'Share Tech Mono', color: '#5f7a9e', size: 10 }
    }] : [],
    scene: {
      bgcolor: 'rgba(0,0,0,0)',
      xaxis: {
        title: { text: 'X', font: { color: '#334762', size: 10 } },
        color: '#334762', gridcolor: '#192232', zerolinecolor: '#243448',
        showbackground: true, backgroundcolor: 'rgba(0,0,0,0.18)'
      },
      yaxis: {
        title: { text: 'Y', font: { color: '#334762', size: 10 } },
        color: '#334762', gridcolor: '#192232', zerolinecolor: '#243448',
        showbackground: true, backgroundcolor: 'rgba(0,0,0,0.12)'
      },
      zaxis: {
        title: { text: 'Z (Depth)', font: { color: '#334762', size: 10 } },
        color: '#334762', gridcolor: '#192232', zerolinecolor: '#243448',
        showbackground: true, backgroundcolor: 'rgba(0,0,0,0.08)'
      },
      camera: {
        eye: { x: 1.5, y: 1.5, z: 0.75 },
        center: { x: 0, y: 0, z: 0 }
      }
    },
    legend: {
      font: { family: 'Share Tech Mono', color: '#5f7a9e', size: 9 },
      bgcolor: 'rgba(11,16,25,0.85)',
      bordercolor: '#192232', borderwidth: 1,
      x: 0.01, y: 0.99, xanchor: 'left', yanchor: 'top'
    },
    hoverlabel: {
      bgcolor: '#0b1019',
      bordercolor: '#243448',
      font: { family: 'Share Tech Mono', color: '#8ab6e0', size: 10 }
    }
  };
}

const plotlyConfig = {
  displayModeBar: true,
  modeBarButtonsToRemove: ['toImage', 'sendDataToCloud', 'select3d', 'lasso3d'],
  displaylogo: false,
  responsive: true
};

function renderGazeVisualizer(containerId, coords, vergencePoint, isDeepfake) {
  const subtitle = isDeepfake
    ? '⚠ Gaze rays do not converge — anomaly detected'
    : '✓ Gaze rays converge correctly — geometry valid';

  const traces = buildGazeTraces(coords, vergencePoint);
  const layout = getPlotLayout(subtitle);

  Plotly.newPlot(containerId, traces, layout, plotlyConfig)
    .catch(err => console.error('[SVVAP] Plotly error:', err));
}

// ─── EXPAND MODAL ────────────────────────────────────────────────────────────

btnExpand.addEventListener('click', () => {
  if (!state.result) return;

  modalOverlay.style.display = 'flex';
  document.body.style.overflow = 'hidden';

  const { raw_coordinates, analysis } = state.result;
  const subtitle = analysis.is_deepfake
    ? '⚠ Gaze rays do not converge — anomaly detected'
    : '✓ Gaze rays converge correctly — geometry valid';

  setTimeout(() => {
    const traces = buildGazeTraces(raw_coordinates, analysis.vergence_point);
    const layout = getPlotLayout(subtitle);
    Plotly.newPlot('gazeVisualizerExpanded', traces, layout, plotlyConfig)
      .catch(err => console.error('[SVVAP] Plotly modal error:', err));
  }, 60);
});

function closeModal() {
  modalOverlay.style.display = 'none';
  document.body.style.overflow = '';
  Plotly.purge('gazeVisualizerExpanded');
}

modalClose.addEventListener('click', closeModal);
modalOverlay.addEventListener('click', e => {
  if (e.target === modalOverlay) closeModal();
});
document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && modalOverlay.style.display !== 'none') closeModal();
});

// ─── ANALYZE BUTTON ──────────────────────────────────────────────────────────

btnAnalyze.addEventListener('click', () => {
  if (!btnAnalyze.disabled) analyzeVideo();
});

// ─── DEMO BUTTON ─────────────────────────────────────────────────────────────

btnDemo.addEventListener('click', async () => {
  if (state.phase === 'loading') return;

  // Populate meta as if a file was loaded
  metaName.textContent = 'demo_sample_face.mp4';
  metaSize.textContent = '14.2 MB';
  metaType.textContent = 'video/mp4';
  videoPreviewWrap.classList.remove('visible');
  btnAnalyze.disabled = true;
  btnDemo.disabled = true;

  showPhase('loading');
  setMetaStatus('ANALYZING...', 'analyzing');
  runLoadingSteps();
  videoScanOverlay.classList.add('scanning');

  await new Promise(r => setTimeout(r, 2800));

  clearLoadingSteps();
  videoScanOverlay.classList.remove('scanning');

  state.result = MOCK_RESPONSE;
  state.analysisCount++;
  if (MOCK_RESPONSE.analysis.is_deepfake) state.fakeCount++;
  updateHeaderStats();

  setMetaStatus('DEEPFAKE DETECTED', 'done-fake');
  renderResults(MOCK_RESPONSE);
  showPhase('results');
  btnDemo.disabled = false;
});

// ─── INIT ────────────────────────────────────────────────────────────────────

(function init() {
  showPhase('idle');
  updateHeaderStats();

  console.log('%c SVVAP ', 'background:#00f5d4;color:#070b14;font-family:monospace;font-weight:bold;padding:2px 6px;');
  console.log('%c Spatial Visual Verification & Analysis Platform', 'color:#5f7a9e;font-family:monospace;');
  console.log('%c API: ' + API_ENDPOINT, 'color:#334762;font-family:monospace;');
  console.log('%c Mock mode: ' + USE_MOCK_DATA, 'color:#334762;font-family:monospace;');
})();