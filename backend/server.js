// server.js – video-tool.pro
// -----------------------------------------
// 1. Resolves short‑links (vt.tiktok.com, youtu.be, etc.) before handing them to yt‑dlp.
// 2. Streams the resulting MP4 back to the client and deletes the temp file afterwards.
// 3. Bubbles up yt‑dlp stderr so the API returns useful error messages.
// test test test test test


import express from "express";
import cors from "cors";
import { spawn } from "child_process";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import compression from "express-compression";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Store download status - using videoId as key
const downloads = new Map();

const app = express();
const PORT = process.env.PORT || 3000;

// Enable compression for all routes (reduces payload size)
app.use(compression());

// Set up CORS with specific options for better performance
app.use(cors({
  origin: ['http://localhost:3000', 'http://localhost:5173'],
  methods: ['GET', 'POST'],
  allowedHeaders: ['Content-Type']
}));

app.use(express.json());

// Serve static frontend with general caching headers
const staticOptions = {
  maxAge: '1h',
  etag: true,
  lastModified: true
};

// Apply compression and general static file handling
app.use(express.static(path.join(__dirname, "../frontend"), staticOptions));

// Special route for CSS and JS assets with longer cache time
app.get('/assets/:file', (req, res, next) => {
  const filePath = path.join(__dirname, '../frontend/assets', req.params.file);
  
  // Check if file exists
  if (fs.existsSync(filePath)) {
    // Set aggressive caching for CSS and JS files
    const extension = path.extname(filePath);
    if (extension === '.css' || extension === '.js') {
      res.setHeader('Cache-Control', 'public, max-age=31536000'); // 1 year
      res.setHeader('Expires', new Date(Date.now() + 31536000000).toUTCString());
    }
    // Let the static middleware handle the actual file serving
    next();
  } else {
    next();
  }
});

// Add special route for index.html with shorter cache time
app.get("/", (_, res) => {
  res.setHeader('Cache-Control', 'public, max-age=60'); // 1 minute cache
  res.sendFile(path.join(__dirname, "../frontend/index.html"));
});

// Ensure ./downloads exists
const downloadsDir = path.join(__dirname, "downloads");
if (!fs.existsSync(downloadsDir)) fs.mkdirSync(downloadsDir, { recursive: true });

// Garbage collection for old downloads (run every 10 minutes)
setInterval(() => {
  const now = Date.now();
  let count = 0;
  
  downloads.forEach((download, videoId) => {
    // Remove entries older than 30 minutes
    if (now - parseInt(videoId) > 30 * 60 * 1000) {
      downloads.delete(videoId);
      count++;
      
      // Also delete file if it exists and wasn't already deleted
      if (download.filePath && fs.existsSync(download.filePath)) {
        fs.unlinkSync(download.filePath);
      }
    }
  });
  
  if (count > 0) {
    console.log(`[CLEANUP] Removed ${count} old download entries`);
  }
}, 10 * 60 * 1000);

// ────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────
/**
 * Follow redirects and return the final URL
 */
async function resolveRedirect(rawUrl) {
  try {
    const res = await fetch(rawUrl, { 
      method: "HEAD", 
      redirect: "follow",
      timeout: 5000 // 5 second timeout
    });
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

  // Only log raw output in development environment
  if (process.env.NODE_ENV === 'development') {
    console.log(`[RAW OUTPUT] ${line}`);
  }
  
  let progressMessage = null;
  
  // Process download percentage updates
  if (line.includes('[download]') && line.includes('%')) {
    progressMessage = line.replace(/^\[download\]\s+/, '').trim();
    // Reduce logging verbosity - only log significant progress changes
    if (line.includes('100%') || line.includes('ETA')) {
      console.log(`[PROGRESS] VideoID: ${videoId}, Progress: ${progressMessage}`);
    }
  } 
  // Process download file ready
  else if (line.startsWith(downloadsDir)) {
    progressMessage = "100% - Download complete!";
    console.log(`[COMPLETE] VideoID: ${videoId}, File: ${path.basename(line)}`);
    const currentDownload = downloads.get(videoId) || {};
    downloads.set(videoId, { 
      status: 'complete', 
      message: progressMessage,
      filePath: line,
      timestamp: Date.now(),
      // Preserve url if it exists
      url: currentDownload.url 
    });
    return line; // Return the final path
  }
  // Other informational messages
  else if (line.includes('[download]')) {
    progressMessage = line.replace(/^\[download\]\s+/, '').trim();
    // Only log important informational messages
    if (line.includes('Destination:') || line.includes('error')) {
      console.log(`[INFO] VideoID: ${videoId}, Message: ${progressMessage}`);
    }
  }
  
  // Update the download status if we have a progress message
  if (progressMessage) {
    const currentDownload = downloads.get(videoId) || {};
    downloads.set(videoId, { 
      status: 'downloading',
      message: progressMessage,
      timestamp: Date.now(),
      // Preserve url if it exists
      url: currentDownload.url
    });
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
    
    console.log(`[DOWNLOAD START] VideoID: ${videoId}`);
    
    // Initialize download status - IMPORTANT: Must be set before spawning the process
    downloads.set(videoId, { 
      status: 'starting', 
      message: 'Starting download...',
      timestamp: Date.now(),
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
          timestamp: Date.now(),
          url: currentDownload.url // Preserve the URL
        });
        return reject(new Error(errBuf || "yt-dlp failed"));
      }
      
      const currentDownload = downloads.get(videoId) || {};
      downloads.set(videoId, { 
        status: 'complete', 
        message: '100% complete',
        filePath: finalPath,
        timestamp: Date.now(),
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
  
  if (!videoId) {
    return res.status(400).json({ 
      status: 'error', 
      message: 'VideoID is required' 
    });
  }
  
  // Set cache headers to prevent browser caching
  res.setHeader('Cache-Control', 'no-store, max-age=0');
  
  const download = downloads.get(videoId);
  
  if (!download) {
    // Return a generic "processing" message instead of "not found"
    return res.json({ 
      status: 'processing', 
      message: 'Processing your download...' 
    });
  }
  
  // Create a copy of the download object without the URL for the response
  const downloadResponse = { ...download };
  delete downloadResponse.url; // Remove URL to not expose it to frontend
  delete downloadResponse.timestamp; // Remove timestamp
  
  res.json(downloadResponse);
});

// Start download
app.post("/download", async (req, res) => {
  const { url } = req.body;
  if (!url) return res.status(400).json({ error: "No URL provided" });

  try {
    // Generate a unique ID for this download
    const videoId = Date.now().toString();
    console.log(`[NEW DOWNLOAD] VideoID: ${videoId}`);
    
    // Pre-initialize the download entry to ensure it's there
    downloads.set(videoId, { 
      status: 'initializing', 
      message: 'Initializing download...',
      timestamp: Date.now(),
      url: url // Store the original URL
    });
    
    // Send back the video ID immediately
    res.json({ videoId });

    // Resolve the URL first (this happens after response is sent)
    const resolvedUrl = await resolveRedirect(url);
    
    // Check if the entry is still there
    if (!downloads.has(videoId)) {
      console.error(`[ERROR] VideoID ${videoId} missing from downloads map before download starts`);
      downloads.set(videoId, { 
        status: 'initializing', 
        message: 'Initializing download...',
        timestamp: Date.now(),
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
    console.error(`[DOWNLOAD ERROR]`, err.message);
    const videoId = err.videoId || req.body.videoId;
    if (videoId) {
      const currentDownload = downloads.get(videoId) || {};
      downloads.set(videoId, { 
        status: 'error', 
        message: err.message || "Download failed",
        timestamp: Date.now(),
        url: currentDownload.url // Preserve the URL
      });
    }
  }
});

// Get the file
app.get('/download/:videoId', (req, res) => {
  const videoId = req.params.videoId;
  
  const download = downloads.get(videoId);
  
  if (!download || download.status !== 'complete') {
    return res.status(404).send('Download not found or not complete');
  }
  
  const filePath = download.filePath;
  if (!filePath || !fs.existsSync(filePath)) {
    return res.status(404).send('Download file not found');
  }
  
  const filename = path.basename(filePath);
  const fileStats = fs.statSync(filePath);
  
  // Set appropriate headers for video download
  res.setHeader('Content-Type', 'video/mp4');
  res.setHeader('Content-Disposition', `attachment; filename="${encodeURIComponent(filename)}"`);
  res.setHeader('Content-Length', fileStats.size);
  res.setHeader('Accept-Ranges', 'bytes');
  res.setHeader('Cache-Control', 'private, max-age=31536000'); // 1 year cache for download
  
  // Stream the file efficiently 
  const readStream = fs.createReadStream(filePath);
  readStream.pipe(res);
  
  // Delete file after download
  res.on('finish', () => {
    fs.unlink(filePath, (err) => {
      if (err) console.error(`[CLEANUP ERROR] Error deleting file ${filename}:`, err);
      else console.log(`[CLEANUP] Deleted file ${filename}`);
    });
    
    // Erase the URL immediately after serving the file
    if (downloads.has(videoId)) {
      const currentDownload = downloads.get(videoId);
      delete currentDownload.url; // Erase the URL
      downloads.set(videoId, currentDownload);
    }
    
    // Don't delete the entry right away to allow the frontend to see completion
    setTimeout(() => {
      downloads.delete(videoId);
    }, 60000); // Keep for 1 minute
  });
});

// ────────────────────────────────────────────────────────────
// Start server
// ────────────────────────────────────────────────────────────
app.listen(PORT, () => console.log(`Server running at http://localhost:${PORT}`));
