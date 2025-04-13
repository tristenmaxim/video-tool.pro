// server.js – video-tool.pro
// -----------------------------------------
// 1. Resolves short‑links (vt.tiktok.com, youtu.be, etc.) before handing them to yt‑dlp.
// 2. Streams the resulting MP4 back to the client and deletes the temp file afterwards.
// 3. Bubbles up yt‑dlp stderr so the API returns useful error messages.
// test test test test 


import express from "express";
import cors from "cors";
import { spawn } from "child_process";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Store download status - using videoId as key
const downloads = new Map();

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

// Serve static frontend
app.use(express.static(path.join(__dirname, "../frontend")));
app.get("/", (_, res) => res.sendFile(path.join(__dirname, "../frontend/index.html")));

// Ensure ./downloads exists
const downloadsDir = path.join(__dirname, "downloads");
if (!fs.existsSync(downloadsDir)) fs.mkdirSync(downloadsDir, { recursive: true });

// ────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────
/**
 * Follow redirects and return the final URL
 */
async function resolveRedirect(rawUrl) {
  try {
    const res = await fetch(rawUrl, { method: "HEAD", redirect: "follow" });
    return res.url || rawUrl;
  } catch (err) {
    console.warn("[redirect] fallback to raw URL due to", err.message);
    return rawUrl;
  }
}

// Process each line of yt-dlp output and update progress
function processYtdlpOutput(line, videoId) {
  if (!videoId) {
    console.error("[ERROR] Missing videoId in processYtdlpOutput");
    return null;
  }

  console.log(`[RAW OUTPUT] ${line}`);
  
  let progressMessage = null;
  
  // Process download percentage updates
  if (line.includes('[download]') && line.includes('%')) {
    progressMessage = line.replace(/^\[download\]\s+/, '').trim();
    console.log(`[PROGRESS UPDATE] VideoID: ${videoId}, Progress: ${progressMessage}`);
  } 
  // Process download file ready
  else if (line.startsWith(downloadsDir)) {
    progressMessage = "100% - Download complete!";
    console.log(`[DOWNLOAD COMPLETE] VideoID: ${videoId}, File: ${line}`);
    const currentDownload = downloads.get(videoId) || {};
    downloads.set(videoId, { 
      status: 'complete', 
      message: progressMessage,
      filePath: line,
      // Preserve url if it exists
      url: currentDownload.url 
    });
    return line; // Return the final path
  }
  // Other informational messages
  else if (line.includes('[download]')) {
    progressMessage = line.replace(/^\[download\]\s+/, '').trim();
    console.log(`[INFO] VideoID: ${videoId}, Message: ${progressMessage}`);
  }
  
  // Update the download status if we have a progress message
  if (progressMessage) {
    const currentDownload = downloads.get(videoId) || {};
    downloads.set(videoId, { 
      status: 'downloading',
      message: progressMessage,
      // Preserve url if it exists
      url: currentDownload.url
    });
    
    // Debug log the current download state
    console.log(`[CURRENT STATE] VideoID: ${videoId}:`, downloads.get(videoId));
  }
  
  return null;
}

/**
 * Run yt‑dlp and resolve with the final file path.
 */
function downloadVideo(finalUrl, videoId) {
  return new Promise((resolve, reject) => {
    if (!videoId) {
      return reject(new Error("VideoID is required"));
    }
    
    console.log(`[DOWNLOAD START] Starting download for VideoID: ${videoId}, URL: ${finalUrl}`);
    
    // Initialize download status - IMPORTANT: Must be set before spawning the process
    downloads.set(videoId, { 
      status: 'starting', 
      message: 'Starting download...',
      url: finalUrl  // Store the URL in the download object
    });
    
    const outputTemplate = path.join(downloadsDir, "%(title)s.%(ext)s");

    const args = [
      "-f",
      "bestvideo[ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/best[ext=mp4][vcodec^=avc1]/best",
      "--merge-output-format",
      "mp4",
      "-o",
      outputTemplate,
      "--print",
      "after_move:%(filepath)s",
      "--progress",
      finalUrl,
    ];

    const proc = spawn("yt-dlp", args);
    let finalPath = "";
    let errBuf = "";

    proc.stdout.on("data", (chunk) => {
      const lines = chunk.toString().trim().split('\n');
      for (const line of lines) {
        if (line.trim()) {
          const path = processYtdlpOutput(line, videoId);
          if (path) finalPath = path;
        }
      }
    });

    proc.stderr.on("data", (chunk) => {
      const lines = chunk.toString().trim().split('\n');
      for (const line of lines) {
        if (line.trim()) {
          processYtdlpOutput(line, videoId);
        }
      }
      errBuf += chunk.toString();
    });

    proc.on("close", (code) => {
      if (code !== 0 || !finalPath) {
        const currentDownload = downloads.get(videoId) || {};
        downloads.set(videoId, { 
          status: 'error', 
          message: errBuf || "Download failed",
          url: currentDownload.url // Preserve the URL
        });
        return reject(new Error(errBuf || "yt-dlp failed"));
      }
      
      const currentDownload = downloads.get(videoId) || {};
      downloads.set(videoId, { 
        status: 'complete', 
        message: '100% complete',
        filePath: finalPath,
        url: currentDownload.url // Preserve the URL
      });
      
      resolve(finalPath);
    });
  });
}

// ────────────────────────────────────────────────────────────
// Routes
// ────────────────────────────────────────────────────────────
// Get progress
app.get('/progress/:videoId', (req, res) => {
  const videoId = req.params.videoId;
  console.log(`[PROGRESS REQUEST] VideoID: ${videoId}`);
  
  if (!videoId) {
    return res.status(400).json({ 
      status: 'error', 
      message: 'VideoID is required' 
    });
  }
  
  const download = downloads.get(videoId);
  
  if (!download) {
    console.log(`[PROGRESS NOT FOUND] VideoID: ${videoId} not found in downloads map. Current downloads: ${Array.from(downloads.keys()).join(', ')}`);
    // Return a generic "processing" message instead of "not found" to avoid confusing the user
    return res.json({ 
      status: 'processing', 
      message: 'Processing your download...' 
    });
  }
  
  // Create a copy of the download object without the URL for the response
  const downloadResponse = { ...download };
  delete downloadResponse.url; // Remove URL to not expose it to frontend
  
  console.log(`[PROGRESS RESPONSE] VideoID: ${videoId}, Data:`, downloadResponse);
  res.json(downloadResponse);
});

// Start download
app.post("/download", async (req, res) => {
  const { url } = req.body;
  if (!url) return res.status(400).json({ error: "No URL provided" });

  try {
    // Generate a unique ID for this download
    const videoId = Date.now().toString();
    console.log(`[NEW DOWNLOAD] Creating download with VideoID: ${videoId}, URL: ${url}`);
    
    // Pre-initialize the download entry to ensure it's there
    downloads.set(videoId, { 
      status: 'initializing', 
      message: 'Initializing download...',
      url: url // Store the original URL
    });
    
    // Send back the video ID immediately
    res.json({ videoId });

    // Resolve the URL first (this happens after response is sent)
    const resolvedUrl = await resolveRedirect(url);
    console.log(`[URL RESOLVED] VideoID: ${videoId}, ResolvedURL: ${resolvedUrl}`);
    
    // Check if the entry is still there
    if (!downloads.has(videoId)) {
      console.error(`[ERROR] VideoID ${videoId} missing from downloads map before download starts`);
      downloads.set(videoId, { 
        status: 'initializing', 
        message: 'Initializing download...',
        url: resolvedUrl // Store the resolved URL
      });
    } else {
      // Update with resolved URL
      const currentDownload = downloads.get(videoId);
      downloads.set(videoId, {
        ...currentDownload,
        url: resolvedUrl // Update with resolved URL
      });
    }
    
    // Start the download
    await downloadVideo(resolvedUrl, videoId);
    console.log(`[DOWNLOAD FINISHED] VideoID: ${videoId}`);
  } catch (err) {
    console.error(`[DOWNLOAD ERROR] Error:`, err);
    const videoId = err.videoId || req.body.videoId;
    if (videoId) {
      const currentDownload = downloads.get(videoId) || {};
      downloads.set(videoId, { 
        status: 'error', 
        message: err.message || "Download failed",
        url: currentDownload.url // Preserve the URL
      });
    }
  }
});

// Get the file
app.get('/download/:videoId', (req, res) => {
  const videoId = req.params.videoId;
  console.log(`[FILE REQUEST] VideoID: ${videoId}`);
  
  const download = downloads.get(videoId);
  
  if (!download || download.status !== 'complete') {
    return res.status(404).send('Download not found or not complete');
  }
  
  const filePath = download.filePath;
  if (!filePath || !fs.existsSync(filePath)) {
    return res.status(404).send('Download file not found');
  }
  
  const filename = path.basename(filePath);
  
  res.setHeader('Content-Type', 'video/mp4');
  res.setHeader('Content-Disposition', `attachment; filename="${encodeURIComponent(filename)}"`);
  
  const readStream = fs.createReadStream(filePath);
  readStream.pipe(res);
  
  // Delete file after download
  res.on('finish', () => {
    fs.unlink(filePath, (err) => {
      if (err) console.error(`[CLEANUP ERROR] Error deleting file ${filePath}:`, err);
      else console.log(`[CLEANUP] Deleted file ${filePath}`);
    });
    
    // Erase the URL immediately after serving the file
    if (downloads.has(videoId)) {
      const currentDownload = downloads.get(videoId);
      delete currentDownload.url; // Erase the URL
      downloads.set(videoId, currentDownload);
      console.log(`[SECURITY] Erased URL for VideoID: ${videoId}`);
    }
    
    // Don't delete the entry right away to allow the frontend to see completion
    setTimeout(() => {
      downloads.delete(videoId);
      console.log(`[CLEANUP] Deleted entry for VideoID: ${videoId}`);
    }, 60000); // Keep for 1 minute
  });
});

// ────────────────────────────────────────────────────────────
// Start server
// ────────────────────────────────────────────────────────────
app.listen(PORT, () => console.log(`Server running at http://localhost:${PORT}`));
