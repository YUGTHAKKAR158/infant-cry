/**
 * Infant Cry Diagnostic System JavaScript Controller
 */

let currentObjectUrl = null;

document.addEventListener('DOMContentLoaded', () => {
  loadSampleButtons();
  setupUploadHandlers();
});

const TITLE_MAP = {
  hungry: 'Hungry (Nutritional Demand)',
  belly_pain: 'Belly Pain / Colic',
  burping: 'Burping Needed',
  discomfort: 'Physical Discomfort',
  tired: 'Fatigue / Overtired',
  non_baby_cry: 'Non-Baby Audio'
};

// Fetch preset sample audio files
async function loadSampleButtons() {
  const container = document.getElementById('sample-list');
  if (!container) return;

  try {
    const res = await fetch('/api/samples');
    const data = await res.json();

    container.innerHTML = '';
    data.samples.forEach((sample, index) => {
      const btn = document.createElement('button');
      btn.className = 'sample-btn';
      btn.dataset.relPath = sample.rel_path;
      btn.dataset.fileName = sample.file_name;
      btn.dataset.className = sample.class_name;
      btn.innerHTML = `
        <span>${TITLE_MAP[sample.class_name] || sample.class_name}</span>
        <span class="tag">Sample</span>
      `;
      btn.onclick = () => selectSample(sample, btn);
      container.appendChild(btn);

      // Auto-predict first sample on load
      if (index === 0) {
        selectSample(sample, btn);
      }
    });
  } catch (err) {
    console.error('Failed to load preset samples:', err);
  }
}

// Select and evaluate preset sample
function selectSample(sample, buttonEl) {
  // Highlight active button
  document.querySelectorAll('.sample-btn').forEach(b => b.classList.remove('active'));
  if (buttonEl) buttonEl.classList.add('active');

  // Update File Info Card & Audio Player
  updateActiveFileInfo(sample.file_name, 'Preset Sample Clip', '/' + sample.rel_path);
  updateDropzoneText(`📄 ${sample.file_name}`, 'Preset sample clip loaded • Click to replace file');

  // Trigger classification API
  predictSample(sample.rel_path, sample.file_name);
}

// Handle file drop & upload
function setupUploadHandlers() {
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');

  if (!dropzone || !fileInput) return;

  dropzone.onclick = () => fileInput.click();

  dropzone.ondragover = (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  };

  dropzone.ondragleave = () => {
    dropzone.classList.remove('dragover');
  };

  dropzone.ondrop = (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      handleUserFile(e.dataTransfer.files[0]);
    }
  };

  fileInput.onchange = () => {
    if (fileInput.files.length > 0) {
      handleUserFile(fileInput.files[0]);
    }
  };
}

// Process user-uploaded file
function handleUserFile(file) {
  // De-select sample buttons
  document.querySelectorAll('.sample-btn').forEach(b => b.classList.remove('active'));

  // Clean up previous blob URL if needed
  if (currentObjectUrl) {
    URL.revokeObjectURL(currentObjectUrl);
  }
  currentObjectUrl = URL.createObjectURL(file);

  const sizeKb = (file.size / 1024).toFixed(1);
  updateActiveFileInfo(file.name, `Uploaded File • ${sizeKb} KB`, currentObjectUrl);
  updateDropzoneText(`🎵 ${file.name}`, `File uploaded (${sizeKb} KB) • Click or drag to change`);

  uploadFile(file, file.name);
}

// Update Active File Details UI
function updateActiveFileInfo(fileName, tagText, audioSrc) {
  const nameEl = document.getElementById('active-file-name');
  const tagEl = document.getElementById('active-file-tag');
  const player = document.getElementById('audio-player');
  const sourceNameEl = document.getElementById('source-filename');

  if (nameEl) nameEl.innerText = fileName;
  if (tagEl) tagEl.innerText = tagText;
  if (sourceNameEl) sourceNameEl.innerText = fileName;

  if (player && audioSrc) {
    player.src = audioSrc;
    player.style.display = 'block';
  }
}

// Update Dropzone Status Text
function updateDropzoneText(title, subtitle) {
  const titleEl = document.getElementById('upload-status-title');
  const subEl = document.getElementById('upload-status-sub');
  if (titleEl) titleEl.innerText = title;
  if (subEl) subEl.innerText = subtitle;
}

// Predict via sample path
async function predictSample(relPath, fileName) {
  const formData = new FormData();
  formData.append('sample_path', relPath);

  showLoadingState();

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      body: formData
    });
    const data = await res.json();
    renderDiagnosis(data, fileName);
  } catch (err) {
    console.error('Prediction failed:', err);
  }
}

// Predict via file upload
async function uploadFile(file, fileName) {
  const formData = new FormData();
  formData.append('file', file);

  showLoadingState();

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      body: formData
    });
    const data = await res.json();
    renderDiagnosis(data, fileName);
  } catch (err) {
    console.error('Upload prediction failed:', err);
  }
}

function showLoadingState() {
  const bannerText = document.getElementById('diag-title');
  if (bannerText) bannerText.innerText = 'Analyzing acoustic signal...';
}

// Render diagnosis & analytics onto UI
function renderDiagnosis(data, fallbackFileName) {
  const pred = data.prediction;
  document.getElementById('diag-title').innerText = TITLE_MAP[pred] || pred.toUpperCase();
  document.getElementById('conf-val').innerText = `${data.confidence_pct}%`;

  // Source filename badge
  const sourceFile = data.file_name || fallbackFileName || 'Unknown file';
  const sourceTag = data.source_type || 'Active Audio';
  const sourceNameEl = document.getElementById('source-filename');
  if (sourceNameEl) sourceNameEl.innerText = `${sourceFile} (${sourceTag})`;

  // Ensure Audio Player uses backend pre-processed WAV for 100% playable format
  const player = document.getElementById('audio-player');
  if (player && data.audio_url) {
    player.src = data.audio_url;
    player.style.display = 'block';
    player.load();
  }

  // Update Caregiver Recommendation
  document.getElementById('action-text').innerText = data.recommended_action;

  // Update Progress Bars
  for (const [cls, pct] of Object.entries(data.probabilities)) {
    const fillEl = document.getElementById(`fill-${cls}`);
    const valEl = document.getElementById(`val-${cls}`);
    if (fillEl) fillEl.style.width = `${pct}%`;
    if (valEl) valEl.innerText = `${pct}%`;
  }

  // Update Spectrogram Base64
  const specImg = document.getElementById('spectrogram-img');
  if (specImg && data.spectrogram_b64) {
    specImg.src = data.spectrogram_b64;
  }

  // Update Feature Analytics
  if (data.features) {
    document.getElementById('stat-df').innerText = data.features.fractal_dimension_df;
    document.getElementById('stat-cent').innerText = `${data.features.spectral_centroid_hz} Hz`;
  }
}
