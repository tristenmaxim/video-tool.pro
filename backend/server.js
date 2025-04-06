// server.js – DOWNLOAD‑VIDEO.PRO (Node 18+)
// -----------------------------------------
// 1. Resolves short‑links (vt.tiktok.com, youtu.be, etc.) before handing them to yt‑dlp.
// 2. Streams the resulting MP4 back to the client and deletes the temp file afterwards.
// 3. Bubbles up yt‑dlp stderr so the API returns useful error messages.
// test


import express from "express";
import cors from "cors";
import { spawn } from "child_process";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

// ────────────────────────────────────────────────────────────
// Basic setup
// ────────────────────────────────────────────────────────────
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

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
 * Follow redirects and return the final URL (Node 18+ global fetch).
 * HEAD request is enough; it saves bandwidth and is faster.
 */
async function resolveRedirect(rawUrl) {
  try {
    const res = await fetch(rawUrl, { method: "HEAD", redirect: "follow" });
    return res.url || rawUrl;
  } catch (err) {
    console.warn("[redirect] fallback to raw URL due to", err.message);
    return rawUrl; // fall back gracefully
  }
}

/**
 * Run yt‑dlp and resolve with the final file path.
 */
function downloadVideo(finalUrl) {
  return new Promise((resolve, reject) => {
    const outputTemplate = path.join(downloadsDir, "%(title)s.%(ext)s");

    const args = [
      "-f",
      "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
      "--merge-output-format",
      "mp4",
      "-o",
      outputTemplate,
      "--print",
      "after_move:%(filepath)s",
      finalUrl,
    ];

    const proc = spawn("yt-dlp", args);

    let finalPath = "";
    let errBuf = "";

    proc.stdout.on("data", (chunk) => {
      const line = chunk.toString().trim();
      console.log("[yt-dlp]", line);
      if (line.startsWith(downloadsDir)) finalPath = line; // captured from --print
    });

    proc.stderr.on("data", (chunk) => {
      const msg = chunk.toString();
      errBuf += msg;
      console.error("[yt-dlp]", msg);
    });

    proc.on("close", (code) => {
      if (code !== 0 || !finalPath) {
        return reject(new Error(errBuf || "yt-dlp failed"));
      }
      resolve(finalPath);
    });
  });
}

// ────────────────────────────────────────────────────────────
// Routes
// ────────────────────────────────────────────────────────────
app.post("/download", async (req, res) => {
  const { url } = req.body;
  if (!url) return res.status(400).json({ error: "No URL provided" });

  try {
    const resolvedUrl = await resolveRedirect(url);
    const filePath = await downloadVideo(resolvedUrl);
    const filename = path.basename(filePath);

    // Headers for attachment download
    res.setHeader("Content-Type", "video/mp4");
    res.setHeader("Content-Disposition", `attachment; filename=\"${encodeURIComponent(filename)}\"`);
    res.setHeader("Cache-Control", "no-store");

    const readStream = fs.createReadStream(filePath);
    readStream.pipe(res);

    // Delete temp file when response is done
    res.on("finish", () => {
      fs.unlink(filePath, (err) => err && console.error("[cleanup]", err));
    });
  } catch (err) {
    console.error("[download]", err);
    res.status(500).json({ error: err.message || "Download failed" });
  }
});

// ────────────────────────────────────────────────────────────
// Start server
// ────────────────────────────────────────────────────────────
app.listen(PORT, () => console.log(`Server running at http://localhost:${PORT}`));
