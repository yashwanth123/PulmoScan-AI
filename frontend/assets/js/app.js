/**
 * PulmoScan AI — Frontend Application
 */
(() => {
  'use strict';

  const API = '';
  let selectedScanType = 'chest_xray';
  let selectedFile = null;

  const CLASS_COLORS = {
    'COVID-19': '#f87171',
    'Normal': '#34d399',
    'Pneumonia': '#fbbf24',
    'Tuberculosis': '#a78bfa',
  };

  const CLASS_ICONS = {
    'COVID-19': '🦠',
    'Normal': '✅',
    'Pneumonia': '🫁',
    'Tuberculosis': '🔬',
  };

  const SCAN_ICONS = { xray: '🩻', ct: '🖥️', mri: '🧲' };

  // ── DOM refs ──
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  const dropzone = $('#dropzone');
  const fileInput = $('#file-input');
  const previewContainer = $('#preview-container');
  const previewImage = $('#preview-image');
  const dropzoneContent = $('#dropzone-content');
  const analyzeBtn = $('#analyze-btn');
  const clearBtn = $('#clear-upload');
  const loadingOverlay = $('#loading-overlay');
  const resultsEmpty = $('#results-empty');
  const resultsContent = $('#results-content');

  // ── Init ──
  document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initUpload();
    loadScanTypes();
    checkModelStatus();
    refreshDashboard();
  });

  // ── Navigation ──
  function initNavigation() {
    $$('.nav-item').forEach((btn) => {
      btn.addEventListener('click', () => {
        const section = btn.dataset.section;
        $$('.nav-item').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        $$('.section').forEach((s) => s.classList.remove('active'));
        $(`#section-${section}`).classList.add('active');
        $('#page-title').textContent = btn.textContent.trim();
        if (section === 'dashboard') refreshDashboard();
        if (section === 'history') refreshHistory();
        $('#sidebar').classList.remove('open');
      });
    });

    $('#menu-toggle')?.addEventListener('click', () => {
      $('#sidebar').classList.toggle('open');
    });
  }

  // ── Upload ──
  function initUpload() {
    dropzone.addEventListener('click', (e) => {
      if (e.target === clearBtn || clearBtn?.contains(e.target)) return;
      fileInput.click();
    });

    fileInput.addEventListener('change', () => {
      if (fileInput.files[0]) handleFile(fileInput.files[0]);
    });

    ['dragenter', 'dragover'].forEach((evt) => {
      dropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach((evt) => {
      dropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
      });
    });

    dropzone.addEventListener('drop', (e) => {
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith('image/')) handleFile(file);
    });

    clearBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      clearUpload();
    });

    analyzeBtn.addEventListener('click', runAnalysis);
  }

  function handleFile(file) {
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImage.src = e.target.result;
      dropzoneContent.classList.add('hidden');
      previewContainer.classList.remove('hidden');
      analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  function clearUpload() {
    selectedFile = null;
    fileInput.value = '';
    previewImage.src = '';
    dropzoneContent.classList.remove('hidden');
    previewContainer.classList.add('hidden');
    analyzeBtn.disabled = true;
  }

  // ── Scan types ──
  async function loadScanTypes() {
    try {
      const res = await fetch(`${API}/api/scan-types`);
      const data = await res.json();
      const grid = $('#scan-types');
      grid.innerHTML = data.scan_types.map((st) => `
        <div class="scan-type-card ${st.key === selectedScanType ? 'selected' : ''} ${!st.supported ? 'disabled' : ''}"
             data-key="${st.key}" data-supported="${st.supported}">
          <span class="scan-type-badge ${st.supported ? '' : 'soon'}">${st.supported ? 'Active' : 'Soon'}</span>
          <div class="scan-type-icon">${SCAN_ICONS[st.icon] || '🩻'}</div>
          <h4>${st.name}</h4>
          <p>${st.description}</p>
        </div>
      `).join('');

      grid.querySelectorAll('.scan-type-card:not(.disabled)').forEach((card) => {
        card.addEventListener('click', () => {
          selectedScanType = card.dataset.key;
          grid.querySelectorAll('.scan-type-card').forEach((c) => c.classList.remove('selected'));
          card.classList.add('selected');
        });
      });

      // Modalities on dashboard
      const modList = $('#modalities-list');
      if (modList) {
        modList.innerHTML = data.scan_types.map((st) => `
          <div class="modality-item">
            <h4>${SCAN_ICONS[st.icon] || '🩻'} ${st.name}</h4>
            <p>${st.description} · ${st.supported ? '✅ Supported' : '⏳ Coming soon'}</p>
          </div>
        `).join('');
      }
    } catch (err) {
      console.error('Failed to load scan types:', err);
    }
  }

  // ── Model status ──
  async function checkModelStatus() {
    const statusEl = $('#model-status');
    const statAcc = $('#stat-accuracy');
    try {
      const res = await fetch(`${API}/api/model`);
      const data = await res.json();
      const dot = statusEl.querySelector('.status-dot');
      const label = statusEl.querySelector('span');

      if (data.model_exists) {
        dot.className = 'status-dot ready';
        label.textContent = 'Model trained & ready';
        statAcc.textContent = 'Ready';
      } else {
        dot.className = 'status-dot loading';
        label.textContent = 'Pretrained backbone (train for accuracy)';
        statAcc.textContent = 'Demo';
      }
    } catch {
      statusEl.querySelector('span').textContent = 'API offline';
    }
  }

  // ── Analysis ──
  async function runAnalysis() {
    if (!selectedFile) return;

    loadingOverlay.classList.remove('hidden');
    analyzeBtn.disabled = true;

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('scan_type', selectedScanType);
    formData.append('include_gradcam', $('#gradcam-toggle').checked);

    try {
      const res = await fetch(`${API}/api/predict`, { method: 'POST', body: formData });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Prediction failed');
      }
      const data = await res.json();
      displayResults(data);
    } catch (err) {
      alert(`Analysis error: ${err.message}`);
    } finally {
      loadingOverlay.classList.add('hidden');
      analyzeBtn.disabled = false;
    }
  }

  function displayResults(data) {
    resultsEmpty.classList.add('hidden');
    resultsContent.classList.remove('hidden');

    $('#diagnosis-value').textContent = data.diagnosis;
    $('#diagnosis-icon').textContent = CLASS_ICONS[data.diagnosis] || '🔬';
    $('#confidence-bar').style.width = `${(data.confidence * 100).toFixed(1)}%`;
    $('#confidence-text').textContent = `${(data.confidence * 100).toFixed(1)}% confidence`;
    $('#results-timestamp').textContent = new Date(data.timestamp).toLocaleString();

    const riskBadge = $('#risk-badge');
    riskBadge.textContent = data.risk_level.replace('-', ' ');
    riskBadge.className = `risk-badge risk-${data.risk_level}`;

    // Probabilities
    const probsEl = $('#probabilities');
    probsEl.innerHTML = Object.entries(data.probabilities)
      .sort((a, b) => b[1] - a[1])
      .map(([label, prob]) => `
        <div class="prob-row">
          <span class="prob-label">${CLASS_ICONS[label] || ''} ${label}</span>
          <div class="prob-track">
            <div class="prob-fill" style="width:${(prob * 100).toFixed(1)}%;background:${CLASS_COLORS[label] || '#38bdf8'}"></div>
          </div>
          <span class="prob-pct">${(prob * 100).toFixed(1)}%</span>
        </div>
      `).join('');

    $('#recommendation-text').textContent = data.recommendation;

    const gradcamSection = $('#gradcam-section');
    if (data.gradcam_image) {
      $('#gradcam-image').src = data.gradcam_image;
      gradcamSection.classList.remove('hidden');
    } else {
      gradcamSection.classList.add('hidden');
    }

    if (!data.model_loaded) {
      $('#recommendation-text').textContent +=
        ' Note: Model has not been fine-tuned yet — run training for clinical-grade accuracy.';
    }
  }

  // ── Dashboard ──
  async function refreshDashboard() {
    try {
      const res = await fetch(`${API}/api/stats`);
      const data = await res.json();

      $('#dash-total').textContent = data.total_scans;
      $('#dash-normal').textContent = data.by_class['Normal'] || 0;
      $('#dash-covid').textContent = data.by_class['COVID-19'] || 0;
      $('#dash-pneumonia').textContent = data.by_class['Pneumonia'] || 0;

      const max = Math.max(...Object.values(data.by_class), 1);
      const chart = $('#breakdown-chart');
      chart.innerHTML = Object.entries(data.by_class).map(([name, count]) => `
        <div class="breakdown-row">
          <span class="breakdown-name">${CLASS_ICONS[name] || ''} ${name}</span>
          <div class="breakdown-bar-wrap">
            <div class="breakdown-bar" style="width:${(count / max) * 100}%;background:${CLASS_COLORS[name] || '#38bdf8'}">
              ${count > 0 ? count : ''}
            </div>
          </div>
        </div>
      `).join('');
    } catch (err) {
      console.error('Dashboard refresh failed:', err);
    }
  }

  // ── History ──
  async function refreshHistory() {
    try {
      const res = await fetch(`${API}/api/history?limit=30`);
      const data = await res.json();
      const list = $('#history-list');

      if (!data.history.length) {
        list.innerHTML = '<p class="empty-state">No analyses yet. Upload a scan to get started.</p>';
        return;
      }

      list.innerHTML = data.history.map((h) => `
        <div class="history-item">
          <span class="history-diagnosis">${CLASS_ICONS[h.diagnosis] || ''} ${h.diagnosis}</span>
          <span class="history-meta">${h.scan_type} · ${new Date(h.timestamp).toLocaleString()}</span>
          <span class="history-confidence">${(h.confidence * 100).toFixed(1)}%</span>
          <span class="risk-badge risk-${h.risk_level}" style="font-size:0.65rem;padding:0.2rem 0.6rem">${h.risk_level}</span>
        </div>
      `).join('');
    } catch (err) {
      console.error('History refresh failed:', err);
    }
  }
})();
