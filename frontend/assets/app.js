// DOM elements - Cache selectors for better performance
const urlInput = document.getElementById('url-input');
const downloadButton = document.getElementById('download-button');
const buttonContainer = document.getElementById('button-container');
const terminalOutput = document.getElementById('terminal-output');
const progressText = document.getElementById('progress-text');
const errorMessage = document.getElementById('error-message');

// Current download state
let currentVideoId = null;
let pollInterval = null;
let pollTimeoutId = null;

// Configure polling with exponential backoff
const POLL_CONFIG = {
  initialInterval: 300,
  maxInterval: 2000,
  currentInterval: 300,
  backoffFactor: 1.5
};

// Add event listener to download button
downloadButton.addEventListener('click', handleDownload);

// Preconnect to API when user focuses on input field
urlInput.addEventListener('focus', () => {
  const link = document.createElement('link');
  link.rel = 'preconnect';
  link.href = 'http://localhost:3000';
  document.head.appendChild(link);
});

async function handleDownload() {
  // Reset UI
  errorMessage.textContent = '';
  errorMessage.classList.add('hidden');
  progressText.textContent = 'Starting download... wait a min';
  
  const url = urlInput.value.trim();
  if (!url) {
    showError('ENTER A URL FIRST');
    return;
  }
  
  // Show terminal output, disable button
  terminalOutput.classList.remove('hidden');
  downloadButton.disabled = true;
  downloadButton.classList.add('btn-disabled');
  downloadButton.classList.remove('bg-white', 'hover:bg-gray-300');
  
  try {
    // Start download
    const res = await fetch('http://localhost:3000/download', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url }),
    });
    
    if (!res.ok) {
      throw new Error(await res.text() || 'Download failed');
    }
    
    const data = await res.json();
    currentVideoId = data.videoId;
    
    // Reset polling configuration
    POLL_CONFIG.currentInterval = POLL_CONFIG.initialInterval;
    
    // Start polling with initial interval
    if (pollInterval) clearInterval(pollInterval);
    if (pollTimeoutId) clearTimeout(pollTimeoutId);
    
    checkProgress();
  } catch (err) {
    console.error('Error starting download:', err);
    showError(err.message || 'Download failed');
    resetUI();
  }
}

async function checkProgress() {
  if (!currentVideoId) return;
  
  try {
    const res = await fetch(`http://localhost:3000/progress/${currentVideoId}`);
    if (!res.ok) {
      console.error('Progress response not OK:', res.status);
      scheduleNextPoll();
      return;
    }
    
    const data = await res.json();
    
    if (data.message) {
      progressText.textContent = data.message;
    }
    
    if (data.status === 'complete') {
      progressText.textContent = '100% complete - Starting download...';
      
      // Download the file using a temporary link element
      const downloadLink = document.createElement('a');
      downloadLink.href = `http://localhost:3000/download/${currentVideoId}`;
      downloadLink.style.display = 'none';
      document.body.appendChild(downloadLink);
      downloadLink.click();
      document.body.removeChild(downloadLink);
      
      // Reset UI after a delay
      setTimeout(resetUI, 3000);
    } else if (data.status === 'error') {
      showError(data.message || 'Download failed');
      resetUI();
    } else {
      // Continue polling with adaptive interval
      scheduleNextPoll();
    }
  } catch (err) {
    console.error('Error checking progress:', err);
    scheduleNextPoll();
  }
}

function scheduleNextPoll() {
  if (pollTimeoutId) clearTimeout(pollTimeoutId);
  
  pollTimeoutId = setTimeout(() => {
    checkProgress();
    
    // Increase polling interval for future polls (exponential backoff)
    POLL_CONFIG.currentInterval = Math.min(
      POLL_CONFIG.currentInterval * POLL_CONFIG.backoffFactor,
      POLL_CONFIG.maxInterval
    );
  }, POLL_CONFIG.currentInterval);
}

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.classList.remove('hidden');
}

function resetUI() {
  terminalOutput.classList.add('hidden');
  downloadButton.disabled = false;
  downloadButton.classList.remove('btn-disabled');
  downloadButton.classList.add('bg-white', 'hover:bg-gray-300');
  currentVideoId = null;
  urlInput.value = '';
  
  // Clear all timers
  if (pollInterval) {
    clearInterval(pollInterval);
    pollInterval = null;
  }
  if (pollTimeoutId) {
    clearTimeout(pollTimeoutId);
    pollTimeoutId = null;
  }
} 