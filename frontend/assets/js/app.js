/**
 * PulmoScan AI — Premium Frontend
 */
(() => {
  'use strict';

  const API = '';
  let selectedScanType = 'chest_xray';
  let selectedFile = null;
  let sampleCache = {};

  const CLASS_COLORS = {
    'COVID-19': '#f87171',
    'Normal': '#34d399',
    'Pneumonia': '#fbbf24',
    'Tuberculosis': '#a855f7',
  };

  const CLASS_ICONS = {
    'COVID-19': '🦠',
    'Normal': '✅',
    'Pneumonia': '🫁',
    'Tuberculosis': '🔬',
  };

  const MODALITY_ICONS = { xray: '🩻', ct: '🖥️', mri: '🧲' };

  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  document.addEventListener('DOMContentLoaded', () => {
    initParticles();
    initNavigation();
    initUpload();
    initSampleTabs();
    loadScanTypes();
    loadSamples();
    checkModelStatus();
    refreshDashboard();
    injectRingGradient();
  });

  function injectRingGradient() {
    const svg = $('.confidence-ring');
    if (!svg || svg.querySelector('#ringGrad')) return;
    const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
    defs.innerHTML = `<linearGradient id="ringGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#00e5ff"/><stop offset="100%" stop-color="#a855f7"/>
    </linearGradient>`;
    svg.prepend(defs);
  }

  /* ── Particle background ── */
  function initParticles() {
    const canvas = $('#particles');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let w, h, dots = [];

    function resize() {
      w = canvas.width = window.innerWidth;
      h = canvas.height = window.innerHeight;
    }

    function init() {
      dots = Array.from({ length: 60 }, () => ({
        x: Math.random() * w,
        y: Math.random() * h,
        r: Math.random() * 1.5 + 0.5,
        dx: (Math.random() - 0.5) * 0.3,
        dy: (Math.random() - 0.5) * 0.3,
      }));
    }

    function draw() {
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = 'rgba(0, 229, 255, 0.4)';
      dots.forEach((d) => {
        d.x += d.dx; d.y += d.dy;
        if (d.x < 0 || d.x > w) d.dx *= -1;
        if (d.y < 0 || d.y > h) d.dy *= -1;
        ctx.beginPath();
        ctx.arc(d.x, d.y, d.r, 0, Math.PI * 2);
        ctx.fill();
      });
      requestAnimationFrame(draw);
    }

    resize();
    init();
    draw();
    window.addEventListener('resize', () => { resize(); init(); });
  }

  /* ── Navigation ── */
  function initNavigation() {
    $$('.nav-item').forEach((btn) => {
      btn.addEventListener('click', () => {
        const section = btn.dataset.section;
        $$('.nav-item').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        $$('.section').forEach((s) => s.classList.remove('active'));
        $(`#section-${section}`)?.classList.add('active');
        $('#page-title').textContent = btn.textContent.trim();
        if (section === 'dashboard') refreshDashboard();
        if (section === 'history') refreshHistory();
        $('#sidebar')?.classList.remove('open');
      });
    });
    $('#menu-toggle')?.addEventListener('click', () => $('#sidebar')?.classList.toggle('open'));
  }

  /* ── Upload ── */
  function initUpload() {
    const dropzone = $('#dropzone');
    const fileInput = $('#file-input');

    dropzone?.addEventListener('click', (e) => {
      if (e.target?.id === 'clear-upload' || e.target?.closest('#clear-upload')) return;
      fileInput?.click();
    });

    fileInput?.addEventListener('change', () => {
      if (fileInput.files?.[0]) handleFile(fileInput.files[0]);
    });

    ['dragenter', 'dragover'].forEach((evt) => {
      dropzone?.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
    });
    ['dragleave', 'drop'].forEach((evt) => {
      dropzone?.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.remove('dragover'); });
    });
    dropzone?.addEventListener('drop', (e) => {
      const file = e.dataTransfer?.files?.[0];
      if (file?.type.startsWith('image/')) handleFile(file);
    });

    $('#clear-upload')?.addEventListener('click', (e) => { e.stopPropagation(); clearUpload(); });
    $('#analyze-btn')?.addEventListener('click', runAnalysis);
  }

  function handleFile(file, scanType) {
    selectedFile = file;
    if (scanType) selectedScanType = scanType;
    updatePreviewTag();

    const reader = new FileReader();
    reader.onload = (e) => {
      const img = $('#preview-image');
      if (img) img.src = e.target.result;
      $('#dropzone-content')?.classList.add('hidden');
      $('#preview-container')?.classList.remove('hidden');
      const btn = $('#analyze-btn');
      if (btn) btn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  function clearUpload() {
    selectedFile = null;
    const fi = $('#file-input');
    if (fi) fi.value = '';
    const img = $('#preview-image');
    if (img) img.src = '';
    $('#dropzone-content')?.classList.remove('hidden');
    $('#preview-container')?.classList.add('hidden');
    const btn = $('#analyze-btn');
    if (btn) btn.disabled = true;
  }

  function updatePreviewTag() {
    const tag = $('#preview-scan-tag');
    if (tag) tag.textContent = selectedScanType === 'ct_scan' ? 'CT Scan' : 'Chest X-Ray';
  }

  /* ── Sample gallery ── */
  function initSampleTabs() {
    $$('.sample-tab').forEach((tab) => {
      tab.addEventListener('click', () => {
        $$('.sample-tab').forEach((t) => t.classList.remove('active'));
        tab.classList.add('active');
        renderSamples(tab.dataset.tab);
      });
    });
  }

  async function loadSamples() {
    try {
      const res = await fetch(`${API}/api/samples`);
      const data = await res.json();
      sampleCache = data.samples || {};
      renderSamples('chest_xray');
    } catch (err) {
      console.error('Samples load failed:', err);
    }
  }

  function renderSamples(scanType) {
    const grid = $('#sample-grid');
    if (!grid) return;
    const items = sampleCache[scanType] || [];

    grid.innerHTML = items.map((s) => `
      <div class="sample-card glass" data-type="${scanType}" data-id="${s.id}">
        <img src="${API}/api/samples/${scanType}/${s.id}" alt="${s.name}" loading="lazy"
             onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22200%22 height=%22120%22><rect fill=%22%231e293b%22 width=%22200%22 height=%22120%22/><text x=%2250%25%22 y=%2250%25%22 fill=%22%2394a3b8%22 text-anchor=%22middle%22 dy=%22.3em%22 font-size=%2214%22>${scanType}</text></svg>'" />
        <div class="sample-card-info">
          <strong>${s.name}</strong>
          <span>${s.label_hint}</span>
        </div>
      </div>
    `).join('');

    grid.querySelectorAll('.sample-card').forEach((card) => {
      card.addEventListener('click', () => loadSample(card.dataset.type, card.dataset.id));
    });
  }

  async function loadSample(scanType, sampleId) {
    try {
      selectedScanType = scanType;
      selectModalityCard(scanType);

      const res = await fetch(`${API}/api/samples/${scanType}/${sampleId}`);
      if (!res.ok) throw new Error('Sample fetch failed');
      const blob = await res.blob();
      const file = new File([blob], `${sampleId}.jpg`, { type: blob.type });
      handleFile(file, scanType);
    } catch (err) {
      alert(`Could not load sample: ${err.message}`);
    }
  }

  function selectModalityCard(key) {
    $$('.modality-card').forEach((c) => {
      c.classList.toggle('selected', c.dataset.key === key);
    });
  }

  /* ── Scan types ── */
  async function loadScanTypes() {
    try {
      const res = await fetch(`${API}/api/scan-types`);
      const data = await res.json();
      const row = $('#scan-types');
      if (!row) return;

      row.innerHTML = data.scan_types.map((st) => `
        <div class="modality-card glass ${st.key === selectedScanType ? 'selected' : ''} ${!st.supported ? 'disabled' : ''}"
             data-key="${st.key}" data-supported="${st.supported}" style="--mod-color:${st.color || '#00e5ff'}">
          <span class="modality-badge ${st.supported ? 'active' : 'soon'}">${st.supported ? 'Active' : 'Soon'}</span>
          <div class="modality-icon-3d">${MODALITY_ICONS[st.icon] || '🩻'}</div>
          <h4>${st.name}</h4>
          <p>${st.description}</p>
        </div>
      `).join('');

      row.querySelectorAll('.modality-card:not(.disabled)').forEach((card) => {
        card.addEventListener('click', () => {
          selectedScanType = card.dataset.key;
          row.querySelectorAll('.modality-card').forEach((c) => c.classList.remove('selected'));
          card.classList.add('selected');
          updatePreviewTag();
        });
      });

      const modList = $('#modalities-list');
      if (modList) {
        modList.innerHTML = data.scan_types.map((st) => `
          <div class="modality-item">
            <h4>${MODALITY_ICONS[st.icon] || '🩻'} ${st.name}</h4>
            <p>${st.description}</p>
          </div>
        `).join('');
      }
    } catch (err) {
      console.error('Scan types failed:', err);
    }
  }

  /* ── Model status ── */
  async function checkModelStatus() {
    const statusEl = $('#model-status');
    const statAcc = $('#stat-accuracy');
    const title = $('#status-title');
    const sub = $('#status-sub');
    try {
      const res = await fetch(`${API}/api/model`);
      const data = await res.json();
      const dot = statusEl?.querySelector('.status-dot');

      if (data.model_exists) {
        dot?.classList.add('ready');
        if (title) title.textContent = 'Model Ready';
        if (sub) sub.textContent = 'Fine-tuned · TTA enabled';
        if (statAcc) statAcc.textContent = 'Trained';
      } else {
        if (title) title.textContent = 'Demo Mode';
        if (sub) sub.textContent = 'Run train.py for accuracy';
        if (statAcc) statAcc.textContent = 'Demo';
      }
    } catch {
      if (title) title.textContent = 'Offline';
    }
  }

  /* ── Analysis ── */
  async function runAnalysis() {
    if (!selectedFile) return;

    $('#loading-overlay')?.classList.remove('hidden');
    $('#analyze-btn').disabled = true;

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('scan_type', selectedScanType);
    formData.append('include_gradcam', $('#gradcam-toggle')?.checked ?? true);

    try {
      const res = await fetch(`${API}/api/predict`, { method: 'POST', body: formData });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Prediction failed');
      }
      displayResults(await res.json());
    } catch (err) {
      alert(`Analysis error: ${err.message}`);
    } finally {
      $('#loading-overlay')?.classList.add('hidden');
      $('#analyze-btn').disabled = false;
    }
  }

  function setConfidenceRing(pct) {
    const ring = $('#confidence-ring');
    const label = $('#ring-pct');
    const circumference = 327;
    const offset = circumference - (pct / 100) * circumference;
    if (ring) ring.style.strokeDashoffset = offset;
    if (label) label.textContent = `${pct.toFixed(1)}%`;
  }

  function displayResults(data) {
    $('#results-empty')?.classList.add('hidden');
    $('#results-content')?.classList.remove('hidden');

    const pct = data.confidence * 100;
    $('#diagnosis-value').textContent = data.diagnosis;
    $('#diagnosis-icon').textContent = CLASS_ICONS[data.diagnosis] || '🔬';
    setConfidenceRing(pct);

    $('#results-timestamp').textContent = new Date(data.timestamp).toLocaleString();

    const riskBadge = $('#risk-badge');
    if (riskBadge) {
      riskBadge.textContent = data.risk_level.replace('-', ' ');
      riskBadge.className = `risk-badge risk-${data.risk_level}`;
    }

    const demoBanner = $('#demo-banner');
    if (demoBanner) demoBanner.classList.toggle('hidden', !data.demo_mode);

    const probsEl = $('#probabilities');
    if (probsEl) {
      probsEl.innerHTML = Object.entries(data.probabilities)
        .sort((a, b) => b[1] - a[1])
        .map(([label, prob]) => {
          const p = prob * 100;
          return `
            <div class="prob-card">
              <div class="prob-card-top">
                <span class="prob-card-label">${CLASS_ICONS[label] || ''} ${label}</span>
                <span class="prob-card-pct" style="color:${CLASS_COLORS[label]}">${p.toFixed(1)}%</span>
              </div>
              <div class="prob-track">
                <div class="prob-fill" style="width:${p}%;background:${CLASS_COLORS[label] || '#00e5ff'}"></div>
              </div>
            </div>`;
        }).join('');
    }

    $('#recommendation-text').textContent = data.recommendation;

    const gradcamSection = $('#gradcam-section');
    if (data.gradcam_image) {
      const img = $('#gradcam-image');
      if (img) img.src = data.gradcam_image;
      gradcamSection?.classList.remove('hidden');
    } else {
      gradcamSection?.classList.add('hidden');
    }
  }

  /* ── Dashboard ── */
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
      if (chart) {
        chart.innerHTML = Object.entries(data.by_class).map(([name, count]) => `
          <div class="breakdown-row">
            <span class="breakdown-name">${CLASS_ICONS[name] || ''} ${name}</span>
            <div class="breakdown-bar-wrap">
              <div class="breakdown-bar" style="width:${(count / max) * 100}%;background:${CLASS_COLORS[name]}">${count || ''}</div>
            </div>
          </div>`).join('');
      }
    } catch (err) {
      console.error('Dashboard failed:', err);
    }
  }

  async function refreshHistory() {
    try {
      const res = await fetch(`${API}/api/history?limit=30`);
      const data = await res.json();
      const list = $('#history-list');
      if (!list) return;

      if (!data.history?.length) {
        list.innerHTML = '<p class="empty-state">No analyses yet.</p>';
        return;
      }

      list.innerHTML = data.history.map((h) => `
        <div class="history-item glass">
          <span class="history-diagnosis">${CLASS_ICONS[h.diagnosis] || ''} ${h.diagnosis}</span>
          <span class="history-meta">${h.scan_type} · ${new Date(h.timestamp).toLocaleString()}</span>
          <span class="history-confidence">${(h.confidence * 100).toFixed(1)}%</span>
          <span class="risk-badge risk-${h.risk_level}" style="font-size:0.6rem;padding:0.2rem 0.5rem">${h.risk_level}</span>
        </div>`).join('');
    } catch (err) {
      console.error('History failed:', err);
    }
  }
})();
