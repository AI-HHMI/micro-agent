"""Web-based dataset explorer with embedded neuroglancer viewer.

Serves a control panel UI alongside an embedded neuroglancer viewer.
Users can select organelle, resolution, repositories, etc. and cycle
through random crops from matching datasets.

Usage:
    pixi run explore
    # or
    python -m trailhead.app [--port 9000] [--ng-port 9001]
"""

from __future__ import annotations

import json
import logging
import os
import socket
import traceback
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

# Suppress tensorstore GCS credential warnings — we use anonymous access
os.environ.setdefault("GOOGLE_AUTH_SUPPRESS_CREDENTIALS_WARNINGS", "true")
os.environ.setdefault("TENSORSTORE_CURL_VERBOSE", "0")

import neuroglancer
import numpy as np
import tornado.gen
import tornado.ioloop
import tornado.web

from trailhead.loader import CropSample, UnifiedLoader
from trailhead.registry import Registry

log = logging.getLogger("trailhead.app")

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_viewer: neuroglancer.Viewer | None = None
_loader: UnifiedLoader | None = None
_loader_iter = None
_current_sample: CropSample | None = None
import queue as _queue_mod
_prefetch_q: _queue_mod.Queue[CropSample | None] = _queue_mod.Queue(maxsize=10)
_prefetch_stop = __import__("threading").Event()
_registry = Registry()
_ng_port = 9001
_crop_count = 0


import re as _re

def _glsl_name(name: str, fallback: str) -> str:
    """Sanitize a channel name into a valid GLSL / neuroglancer invlerp name."""
    # Remove or replace non-alphanumeric characters
    clean = _re.sub(r"[^a-zA-Z0-9_]", "_", name)
    # Strip leading underscores/digits
    clean = clean.lstrip("_0123456789")
    # Collapse multiple underscores
    clean = _re.sub(r"_+", "_", clean).strip("_")
    return clean.lower() or fallback


def _update_viewer(sample: CropSample) -> None:
    """Push a new crop into the neuroglancer viewer."""
    vox = sample.resolution_nm if any(v > 0 for v in sample.resolution_nm) else (1.0, 1.0, 1.0)
    vox_units = ["nm", "nm", "nm"]

    mc = sample.raw_multichannel  # (C, Z, Y, X) or None

    with _viewer.txn() as s:
        s.layers.clear()

        if mc is not None and mc.shape[0] > 1:
            # Multi-channel: display as RGB in neuroglancer
            # Transpose from (C, Z, Y, X) → (Z, Y, X, C) for neuroglancer channel dim
            mc_data = np.transpose(mc, (1, 2, 3, 0))
            cs_mc = neuroglancer.CoordinateSpace(
                names=["z", "y", "x", "c^"],
                units=[*vox_units, ""],
                scales=[vox[0], vox[1], vox[2], 1],
            )

            nch = mc.shape[0]
            n_display = min(nch, 3)

            # Build sanitized unique channel names for invlerp controls
            ch_names = []
            seen: set[str] = set()
            for ci in range(n_display):
                raw = sample.channel_names[ci] if ci < len(sample.channel_names) else f"ch{ci}"
                name = _glsl_name(raw, f"ch{ci}")
                while name in seen:
                    name = f"{name}{ci}"
                seen.add(name)
                ch_names.append(name)

            # Compute per-channel 1-99% ranges
            shader_lines = []
            for ci in range(n_display):
                ch_data = mc[ci]
                p1 = float(np.percentile(ch_data, 1))
                p99 = float(np.percentile(ch_data, 99))
                if p1 == p99:
                    p99 = p1 + 1
                shader_lines.append(
                    f"#uicontrol invlerp {ch_names[ci]}(range=[{p1:.0f}, {p99:.0f}])"
                )

            # Build main() using invlerp(getDataValue(channel))
            if n_display >= 3:
                shader_lines.append("")
                shader_lines.append("void main() {")
                shader_lines.append("  emitRGB(vec3(")
                shader_lines.append(f"    {ch_names[0]}(getDataValue(0)),")
                shader_lines.append(f"    {ch_names[1]}(getDataValue(1)),")
                shader_lines.append(f"    {ch_names[2]}(getDataValue(2))");
                shader_lines.append("  ));")
                shader_lines.append("}")
            elif n_display == 2:
                shader_lines.append("")
                shader_lines.append("void main() {")
                shader_lines.append(f"  float a = {ch_names[0]}(getDataValue(0));")
                shader_lines.append(f"  float b = {ch_names[1]}(getDataValue(1));")
                shader_lines.append("  emitRGB(vec3(b, a, b));")  # green + magenta
                shader_lines.append("}")

            shader = "\n".join(shader_lines)
            s.layers["raw"] = neuroglancer.ImageLayer(
                source=neuroglancer.LocalVolume(data=mc_data, dimensions=cs_mc),
                shader=shader,
            )
        else:
            # Single-channel: grayscale
            cs = neuroglancer.CoordinateSpace(
                names=["z", "y", "x"], units=vox_units, scales=list(vox),
            )
            p1, p99 = float(np.percentile(sample.raw, 1)), float(np.percentile(sample.raw, 99))
            if p1 == p99:
                p99 = p1 + 1
            s.layers["raw"] = neuroglancer.ImageLayer(
                source=neuroglancer.LocalVolume(data=sample.raw, dimensions=cs),
                shader=f"#uicontrol invlerp normalized(range=[{p1:.0f}, {p99:.0f}])\nvoid main() {{\n  emitGrayscale(normalized());\n}}",
            )

        if sample.segmentation is not None:
            cs_seg = neuroglancer.CoordinateSpace(
                names=["z", "y", "x"], units=vox_units, scales=list(vox),
            )
            seg_data = sample.segmentation.astype(np.uint32)
            seg_ids = [int(v) for v in np.unique(seg_data) if v != 0]
            name = (sample.organelle or "seg") + "_seg"
            s.layers[name] = neuroglancer.SegmentationLayer(
                source=neuroglancer.LocalVolume(data=seg_data, dimensions=cs_seg),
                segments=seg_ids,
                selected_alpha=0.3,
            )
        s.position = [sample.raw.shape[i] // 2 for i in range(3)]


def _sample_to_dict(sample: CropSample) -> dict:
    nch = sample.raw_multichannel.shape[0] if sample.raw_multichannel is not None else 1
    return {
        "dataset_id": sample.dataset_id,
        "repository": sample.repository,
        "organelle": sample.organelle,
        "offset": list(sample.offset),
        "resolution_nm": [round(v, 2) for v in sample.resolution_nm],
        "source_resolution_nm": [round(v, 2) for v in sample.source_resolution_nm],
        "scale_used": sample.scale_used,
        "seg_status": sample.seg_status,
        "raw_path": sample.raw_path,
        "seg_path": sample.seg_path,
        "raw_shape": list(sample.raw.shape),
        "raw_min": int(sample.raw.min()),
        "raw_max": int(sample.raw.max()),
        "seg_ids": sorted(int(v) for v in np.unique(sample.segmentation) if v != 0) if sample.segmentation is not None else [],
        "voxel_size_is_estimated": sample.voxel_size_is_estimated,
        "num_channels": nch,
        "channel_names": sample.channel_names,
    }


# ---------------------------------------------------------------------------
# Tornado handlers
# ---------------------------------------------------------------------------

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.set_header("Content-Type", "text/html")
        ng_url = str(_viewer)
        ng_path = urlparse(ng_url).path
        html = HTML_PAGE.replace("{{NG_PORT}}", str(_ng_port)).replace("{{NG_PATH}}", ng_path)
        self.write(html)


class _JsonHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header("Content-Type", "application/json")

    def write_json(self, obj, status=200):
        self.set_status(status)
        self.write(json.dumps(obj))


class OrganellesHandler(_JsonHandler):
    def get(self):
        self.write_json(_registry.list_organelles())


class RepositoriesHandler(_JsonHandler):
    def get(self):
        self.write_json(_registry.list_repositories())


class LoadHandler(_JsonHandler):
    executor = ThreadPoolExecutor(max_workers=2)

    @tornado.gen.coroutine
    def post(self):
        global _loader, _loader_iter, _crop_count
        body = json.loads(self.request.body)
        log.info("Load request: %s", body)

        resolution = body.get("resolution_nm")
        if resolution:
            resolution = tuple(float(v) for v in resolution)

        crop_size = tuple(int(v) for v in body.get("crop_size", [64, 64, 64]))
        repos = body.get("repositories") or None
        if repos is not None and len(repos) == 0:
            repos = None

        # Stop any existing prefetch worker
        _prefetch_stop.set()
        # Drain the queue
        while not _prefetch_q.empty():
            try:
                _prefetch_q.get_nowait()
            except Exception:
                break

        try:
            result = yield self.executor.submit(
                _create_loader,
                organelle=body.get("organelle", ""),
                crop_size=crop_size,
                resolution_nm=resolution,
                require_segmentation=body.get("require_segmentation", False),
                repositories=repos,
                balance_repositories=body.get("balance_repositories", True),
                require_nonempty_raw=body.get("require_nonempty_raw", False),
                require_nonempty_seg=body.get("require_nonempty_seg", False),
                modality_class=body.get("modality_class", ""),
            )
            _loader = result["loader"]
            _loader_iter = iter(_loader)
            _crop_count = 0
            log.info("Loaded %d datasets", result["num_datasets"])

            # Drain queue again to discard anything the old worker snuck in
            while not _prefetch_q.empty():
                try:
                    _prefetch_q.get_nowait()
                except Exception:
                    break

            # Start prefetch worker thread
            _prefetch_stop.clear()
            import threading
            t = threading.Thread(target=_prefetch_worker, daemon=True)
            t.start()

            self.write_json({
                "status": "ok",
                "num_datasets": result["num_datasets"],
                "summary": result["summary"],
            })
        except Exception as e:
            log.error("Load failed: %s\n%s", e, traceback.format_exc())
            self.write_json({"error": str(e)}, status=400)


def _create_loader(**kwargs):
    """Run in thread pool — creates a UnifiedLoader (may do network I/O)."""
    import time
    t0 = time.time()
    loader = UnifiedLoader(num_samples=10000, **kwargs)
    log.info("UnifiedLoader created in %.1fs (%d datasets)", time.time() - t0, len(loader.datasets))
    t1 = time.time()
    summary = loader.summary()
    log.info("loader.summary() in %.1fs", time.time() - t1)
    return {
        "loader": loader,
        "num_datasets": len(loader.datasets),
        "summary": summary,
    }


class NextHandler(_JsonHandler):
    executor = ThreadPoolExecutor(max_workers=2)

    @tornado.gen.coroutine
    def post(self):
        global _current_sample, _crop_count

        if _loader_iter is None:
            self.write_json({"error": "No loader configured. Click Load first."}, status=400)
            return

        try:
            # Block up to 30s waiting for a prefetched crop
            sample = yield self.executor.submit(_get_next_from_queue)
        except Exception as e:
            log.error("Next failed: %s\n%s", e, traceback.format_exc())
            self.write_json({"error": str(e)}, status=500)
            return

        _current_sample = sample
        _crop_count += 1
        _update_viewer(sample)
        result = _sample_to_dict(sample)
        result["crop_number"] = _crop_count
        log.info("Crop #%d: %s (%s)", _crop_count, sample.dataset_id, sample.repository)
        self.write_json(result)


def _get_next_from_queue():
    """Get next crop from the prefetch queue. Blocks until one is available."""
    sample = _prefetch_q.get(timeout=120)
    if sample is None:
        raise RuntimeError("No more samples. Click Load to reset.")
    return sample


def _prefetch_worker():
    """Fill the prefetch queue using a thread pool for parallel fetches.

    Uses multiple threads so a slow dataset (e.g. EMPIAR ~60s) doesn't
    block the queue from being filled by faster sources.
    """
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    NUM_WORKERS = 3
    log.info("Prefetch worker started (%d parallel fetchers)", NUM_WORKERS)

    _iter_lock = threading.Lock()

    def _fetch_next():
        with _iter_lock:
            try:
                return next(_loader_iter)
            except StopIteration:
                return "STOP"

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        # Keep NUM_WORKERS fetches in-flight at all times
        futures = set()
        for _ in range(NUM_WORKERS):
            if not _prefetch_stop.is_set():
                futures.add(pool.submit(_fetch_next))

        while futures and not _prefetch_stop.is_set():
            done = set()
            try:
                for f in as_completed(futures, timeout=1.0):
                    done.add(f)
                    break  # process one at a time to stay responsive to stop
            except TimeoutError:
                continue

            if not done:
                continue

            for f in done:
                futures.discard(f)
                result = f.result()
                if result == "STOP":
                    _prefetch_q.put(None)
                    log.info("Prefetch worker: no more samples")
                    # Cancel remaining futures
                    _prefetch_stop.set()
                    return
                if result is not None:
                    _prefetch_q.put(result)  # blocks if queue full
                    log.info("Prefetched: %s (%s) [queue ~%d]",
                             result.dataset_id, result.repository, _prefetch_q.qsize())

                # Submit a replacement fetch
                if not _prefetch_stop.is_set():
                    futures.add(pool.submit(_fetch_next))

    log.info("Prefetch worker stopped")


# ---------------------------------------------------------------------------
# HTML page
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trailhead Explorer</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

  :root {
    --bg: #0a0e12;
    --panel: #111620;
    --border: #1e2a38;
    --accent: #00d4ff;
    --accent2: #7b61ff;
    --green: #00ff9d;
    --yellow: #ffd166;
    --red: #ff6b6b;
    --text: #c8d8e8;
    --muted: #4a6080;
    --font-mono: 'IBM Plex Mono', monospace;
    --font-sans: 'IBM Plex Sans', sans-serif;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-sans);
    height: 100vh;
    display: grid;
    grid-template-rows: 44px 1fr 0px;
    grid-template-columns: 280px 1fr;
    grid-template-areas:
      "header header"
      "sidebar viewer"
      "meta meta";
    overflow: hidden;
  }

  /* HEADER */
  header {
    grid-area: header;
    background: var(--panel);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    padding: 0 16px;
    gap: 12px;
  }
  .logo {
    font-family: var(--font-mono);
    font-size: 13px;
    font-weight: 600;
    color: var(--accent);
    letter-spacing: 0.08em;
  }
  .logo span { color: var(--muted); }
  #status-text {
    margin-left: auto;
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--muted);
    letter-spacing: 0.05em;
  }

  /* SIDEBAR CONTROLS */
  .sidebar {
    grid-area: sidebar;
    background: var(--panel);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    overflow-y: auto;
    overflow-x: hidden;
  }
  .sidebar::-webkit-scrollbar { width: 4px; }
  .sidebar::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

  .ctrl-section {
    padding: 12px 14px;
    border-bottom: 1px solid var(--border);
  }
  .ctrl-label {
    font-family: var(--font-mono);
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--muted);
    margin-bottom: 8px;
  }

  select, input[type="number"] {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--text);
    font-family: var(--font-mono);
    font-size: 12px;
    padding: 5px 8px;
    outline: none;
    transition: border-color 0.15s;
    width: 100%;
  }
  select:focus, input:focus { border-color: var(--accent); }
  select { cursor: pointer; }

  .triple-input {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 4px;
    align-items: center;
  }
  .triple-input label {
    font-family: var(--font-mono);
    font-size: 9px;
    color: var(--muted);
    text-align: center;
    margin-bottom: 2px;
  }
  .triple-col { display: flex; flex-direction: column; align-items: center; }

  .checkbox-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
    cursor: pointer;
  }
  .checkbox-row input[type="checkbox"] {
    accent-color: var(--accent);
    cursor: pointer;
  }
  .checkbox-row span {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text);
  }

  .repo-checks {
    display: flex;
    flex-direction: column;
    gap: 4px;
    max-height: 180px;
    overflow-y: auto;
  }
  .repo-checks::-webkit-scrollbar { width: 3px; }
  .repo-checks::-webkit-scrollbar-thumb { background: var(--border); }

  .btn {
    width: 100%;
    border: none;
    border-radius: 5px;
    padding: 9px 12px;
    font-family: var(--font-mono);
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }

  .btn-load {
    background: var(--accent);
    color: var(--bg);
    margin-bottom: 6px;
  }
  .btn-load:hover:not(:disabled) { background: #33ddff; }

  .btn-next {
    background: var(--green);
    color: var(--bg);
    font-size: 13px;
    padding: 12px;
  }
  .btn-next:hover:not(:disabled) { background: #33ffb3; }

  .auto-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 8px;
  }
  .auto-row input[type="number"] {
    width: 50px;
    text-align: center;
  }
  .auto-row span {
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--muted);
  }

  /* VIEWER */
  .viewer-area {
    grid-area: viewer;
    position: relative;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }
  #ng-frame {
    flex: 1;
    width: 100%;
    border: none;
    background: var(--bg);
  }
  /* META BAR (inside viewer area, at bottom) */
  .meta-bar {
    background: var(--panel);
    border-bottom: 1px solid var(--border);
    padding: 8px 14px;
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--muted);
    display: none;
    gap: 6px;
    flex-direction: column;
    min-height: 70px;
    overflow-y: auto;
  }
  .meta-bar.visible { display: flex; }
  .meta-row {
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
    line-height: 1.6;
  }
  .meta-key { color: var(--muted); }
  .meta-val { color: var(--text); }
  .meta-val.good { color: var(--green); }
  .meta-val.warn { color: var(--yellow); }
  .meta-val.bad { color: var(--red); }

  /* SUMMARY OVERLAY */
  .summary-box {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 8px 10px;
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--text);
    line-height: 1.6;
    white-space: pre-wrap;
    max-height: 160px;
    overflow-y: auto;
    margin-top: 6px;
  }
  .summary-box::-webkit-scrollbar { width: 3px; }
  .summary-box::-webkit-scrollbar-thumb { background: var(--border); }

  .error-msg {
    background: rgba(255,107,107,0.08);
    border: 1px solid rgba(255,107,107,0.3);
    border-radius: 4px;
    padding: 8px 10px;
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--red);
    margin-top: 6px;
  }
</style>
</head>
<body>

<header>
  <div class="logo">Trail<span>head</span> Explorer</div>
  <span id="status-text">configure + load to start</span>
</header>

<div class="sidebar">
  <!-- Modality -->
  <div class="ctrl-section">
    <div class="ctrl-label">Modality</div>
    <select id="modality">
      <option value="">All (EM + fluorescence)</option>
      <option value="em">EM only</option>
      <option value="fluorescence">Fluorescence only</option>
    </select>
  </div>

  <!-- Organelle -->
  <div class="ctrl-section">
    <div class="ctrl-label">Organelle</div>
    <select id="organelle">
      <option value="">(any / raw only)</option>
    </select>
  </div>

  <!-- Resolution -->
  <div class="ctrl-section">
    <div class="ctrl-label">Resolution (nm)</div>
    <div class="triple-input">
      <div class="triple-col"><label>z</label><input type="number" id="res-z" value="64" min="1" step="1"></div>
      <div class="triple-col"><label>y</label><input type="number" id="res-y" value="64" min="1" step="1"></div>
      <div class="triple-col"><label>x</label><input type="number" id="res-x" value="64" min="1" step="1"></div>
    </div>
    <div class="checkbox-row" style="margin-top:6px">
      <input type="checkbox" id="iso-lock" checked>
      <span>Isotropic lock</span>
    </div>
  </div>

  <!-- Crop size -->
  <div class="ctrl-section">
    <div class="ctrl-label">Crop size (voxels)</div>
    <div class="triple-input">
      <div class="triple-col"><label>z</label><input type="number" id="crop-z" value="64" min="8" step="8"></div>
      <div class="triple-col"><label>y</label><input type="number" id="crop-y" value="64" min="8" step="8"></div>
      <div class="triple-col"><label>x</label><input type="number" id="crop-x" value="64" min="8" step="8"></div>
    </div>
  </div>

  <!-- Options -->
  <div class="ctrl-section">
    <div class="ctrl-label">Options</div>
    <div class="checkbox-row">
      <input type="checkbox" id="require-seg" disabled>
      <span>Require segmentation</span>
    </div>
    <div class="checkbox-row">
      <input type="checkbox" id="balance-repos" checked>
      <span>Balance across repos</span>
    </div>
    <div class="checkbox-row">
      <input type="checkbox" id="require-nonempty-raw" checked>
      <span>Require nonzero raw</span>
    </div>
    <div class="checkbox-row">
      <input type="checkbox" id="require-nonempty-seg" disabled>
      <span>Require nonzero seg</span>
    </div>
  </div>

  <!-- Repositories -->
  <div class="ctrl-section">
    <div class="ctrl-label">Repositories</div>
    <div class="repo-checks" id="repo-checks">
      <!-- populated by JS -->
    </div>
  </div>

  <!-- Actions -->
  <div class="ctrl-section">
    <button class="btn btn-load" id="btn-load" onclick="doLoad()">Load</button>
    <button class="btn btn-next" id="btn-next" onclick="doNext()" disabled>Next Crop</button>
    <div class="auto-row">
      <input type="checkbox" id="auto-cycle">
      <span>Auto every</span>
      <input type="number" id="auto-interval" value="5" min="1" max="60">
      <span>s</span>
    </div>
    <div id="summary-area"></div>
  </div>
</div>

<div class="viewer-area">
  <div class="meta-bar" id="meta-bar">
    <div id="voxel-warning" style="display:none; background:rgba(255,107,107,0.12); border:1px solid rgba(255,107,107,0.4); border-radius:3px; padding:4px 8px; color:var(--red); font-size:10px; line-height:1.4;">
      No voxel size metadata in file — crop is raw voxels (no resampling). Displayed resolution is nominal.
    </div>
    <div class="meta-row" id="meta-row-1"></div>
    <div class="meta-row" id="meta-row-2"></div>
    <div class="meta-row" id="meta-row-3"></div>
  </div>
  <iframe id="ng-frame"></iframe>
</div>

<script>
const NG_PORT = {{NG_PORT}};
const NG_PATH = "{{NG_PATH}}";
let autoTimer = null;
let loaded = false;
let fetching = false;

// --- Init: populate organelles and repos, set iframe src ---
async function init() {
  // Point the iframe at the neuroglancer viewer immediately
  const frame = document.getElementById('ng-frame');
  frame.src = `http://${window.location.hostname}:${NG_PORT}${NG_PATH}`;

  try {
    const [orgRes, repoRes] = await Promise.all([
      fetch('/api/organelles').then(r => r.json()),
      fetch('/api/repositories').then(r => r.json()),
    ]);

    const orgSel = document.getElementById('organelle');
    orgRes.forEach(o => {
      const opt = document.createElement('option');
      opt.value = o;
      opt.textContent = o;
      orgSel.appendChild(opt);
    });

    const repoDiv = document.getElementById('repo-checks');
    repoRes.forEach(r => {
      const row = document.createElement('label');
      row.className = 'checkbox-row';
      row.innerHTML = `<input type="checkbox" value="${r}" checked><span>${r}</span>`;
      repoDiv.appendChild(row);
    });
  } catch (e) {
    console.error('Init failed:', e);
  }
}
init();

// --- Organelle → toggle seg-related checkboxes ---
document.getElementById('organelle').addEventListener('change', function() {
  const reqSeg = document.getElementById('require-seg');
  const segCb = document.getElementById('require-nonempty-seg');
  if (!this.value) {
    reqSeg.checked = false;
    reqSeg.disabled = true;
    segCb.checked = false;
    segCb.disabled = true;
  } else {
    reqSeg.disabled = false;
    segCb.disabled = false;
  }
});

// --- Isotropic lock ---
document.getElementById('res-z').addEventListener('input', function() {
  if (document.getElementById('iso-lock').checked) {
    document.getElementById('res-y').value = this.value;
    document.getElementById('res-x').value = this.value;
  }
});
document.getElementById('res-y').addEventListener('input', function() {
  if (document.getElementById('iso-lock').checked) {
    document.getElementById('res-z').value = this.value;
    document.getElementById('res-x').value = this.value;
  }
});
document.getElementById('res-x').addEventListener('input', function() {
  if (document.getElementById('iso-lock').checked) {
    document.getElementById('res-z').value = this.value;
    document.getElementById('res-y').value = this.value;
  }
});

// --- Load ---
async function doLoad() {
  const btn = document.getElementById('btn-load');
  btn.disabled = true;
  btn.textContent = 'Loading...';
  document.getElementById('summary-area').innerHTML = '';

  const repos = [...document.querySelectorAll('#repo-checks input:checked')].map(c => c.value);

  const body = {
    modality_class: document.getElementById('modality').value,
    organelle: document.getElementById('organelle').value,
    resolution_nm: [
      parseFloat(document.getElementById('res-z').value),
      parseFloat(document.getElementById('res-y').value),
      parseFloat(document.getElementById('res-x').value),
    ],
    crop_size: [
      parseInt(document.getElementById('crop-z').value),
      parseInt(document.getElementById('crop-y').value),
      parseInt(document.getElementById('crop-x').value),
    ],
    require_segmentation: document.getElementById('require-seg').checked,
    balance_repositories: document.getElementById('balance-repos').checked,
    require_nonempty_raw: document.getElementById('require-nonempty-raw').checked,
    require_nonempty_seg: document.getElementById('require-nonempty-seg').checked,
    repositories: repos,
  };

  try {
    const res = await fetch('/api/load', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json();

    if (!res.ok) {
      document.getElementById('summary-area').innerHTML =
        `<div class="error-msg">${data.error}</div>`;
      return;
    }

    loaded = true;
    document.getElementById('btn-next').disabled = false;
    document.getElementById('status-text').textContent =
      `${data.num_datasets} dataset(s) loaded — fetching first crop...`;
    document.getElementById('summary-area').innerHTML =
      `<div class="summary-box">${escHtml(data.summary)}</div>`;

    // Auto-fetch the first crop immediately
    btn.disabled = false;
    btn.textContent = 'Load';
    doNext();
    return;
  } catch (e) {
    document.getElementById('summary-area').innerHTML =
      `<div class="error-msg">Request failed: ${e.message}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = 'Load';
  }
}

// --- Next crop ---
async function doNext() {
  if (fetching || !loaded) return;
  fetching = true;
  const btn = document.getElementById('btn-next');
  btn.disabled = true;
  btn.textContent = 'Fetching...';

  try {
    const res = await fetch('/api/next', { method: 'POST' });
    const data = await res.json();

    if (!res.ok) {
      document.getElementById('status-text').textContent = data.error;
      return;
    }

    // neuroglancer auto-updates via websocket — no iframe reload needed

    // Update metadata
    const mb = document.getElementById('meta-bar');
    mb.classList.add('visible');

    // Voxel size warning
    const voxWarnEl = document.getElementById('voxel-warning');
    voxWarnEl.style.display = data.voxel_size_is_estimated ? 'block' : 'none';

    const segClass = data.seg_status === 'loaded' ? 'good'
                   : data.seg_status === 'empty' ? 'warn'
                   : data.seg_status.startsWith('failed') ? 'bad' : '';

    const chInfo = data.num_channels > 1
      ? `<span><span class="meta-key">channels:</span> <span class="meta-val">${data.num_channels} (${data.channel_names.join(', ') || 'RGB'})</span></span>`
      : '';
    document.getElementById('meta-row-1').innerHTML = `
      <span><span class="meta-key">dataset:</span> <span class="meta-val">${data.dataset_id}</span></span>
      <span><span class="meta-key">repo:</span> <span class="meta-val">${data.repository}</span></span>
      <span><span class="meta-key">crop #</span> <span class="meta-val">${data.crop_number}</span></span>
      ${chInfo}
      <span><span class="meta-key">seg:</span> <span class="meta-val ${segClass}">${data.seg_status}</span></span>
    `;
    const srcText = data.voxel_size_is_estimated
      ? `<span class="meta-val bad">unknown (no metadata)</span>`
      : `<span class="meta-val">${data.source_resolution_nm.join(' x ')} nm @ s${data.scale_used}</span>`;
    document.getElementById('meta-row-2').innerHTML = `
      <span><span class="meta-key">resolution:</span> <span class="meta-val">${data.resolution_nm.join(' x ')} nm</span></span>
      <span><span class="meta-key">source:</span> ${srcText}</span>
      <span><span class="meta-key">offset:</span> <span class="meta-val">(${data.offset.join(', ')})</span></span>
      <span><span class="meta-key">shape:</span> <span class="meta-val">${data.raw_shape.join(' x ')}</span></span>
      <span><span class="meta-key">range:</span> <span class="meta-val">[${data.raw_min}, ${data.raw_max}]</span></span>
    `;
    document.getElementById('meta-row-3').innerHTML = `
      <span><span class="meta-key">raw:</span> <span class="meta-val">${data.raw_path || 'n/a'}</span></span>
      ${data.seg_path ? `<span><span class="meta-key">seg:</span> <span class="meta-val">${data.seg_path}</span></span>` : ''}
    `;

    document.getElementById('status-text').textContent =
      `crop #${data.crop_number} — ${data.dataset_id}`;

  } catch (e) {
    document.getElementById('status-text').textContent = 'Error: ' + e.message;
  } finally {
    fetching = false;
    btn.disabled = false;
    btn.textContent = 'Next Crop';
  }
}

// --- Auto-cycle ---
document.getElementById('auto-cycle').addEventListener('change', function() {
  if (this.checked) {
    const sec = parseInt(document.getElementById('auto-interval').value) || 5;
    autoTimer = setInterval(() => { if (loaded && !fetching) doNext(); }, sec * 1000);
  } else {
    clearInterval(autoTimer);
    autoTimer = null;
  }
});

// --- Keyboard shortcut: space or right-arrow for next ---
document.addEventListener('keydown', function(e) {
  // Don't trigger if typing in an input
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if ((e.key === ' ' || e.key === 'ArrowRight') && loaded) {
    e.preventDefault();
    doNext();
  }
});

function escHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def serve(port: int = 9000, ng_port: int = 9001) -> None:
    """Start the Trailhead web explorer."""
    global _viewer, _ng_port
    _ng_port = ng_port
    hostname = socket.gethostname()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    neuroglancer.set_server_bind_address("0.0.0.0", ng_port)
    _viewer = neuroglancer.Viewer()

    app = tornado.web.Application([
        (r"/", MainHandler),
        (r"/api/organelles", OrganellesHandler),
        (r"/api/repositories", RepositoriesHandler),
        (r"/api/load", LoadHandler),
        (r"/api/next", NextHandler),
    ])
    app.listen(port, "0.0.0.0")

    print(f"\n  Trailhead Explorer:  http://{hostname}:{port}/")
    print(f"  Neuroglancer:        http://{hostname}:{ng_port}/")
    print(f"\n  Press Ctrl+C to stop.\n")

    # Warm up heavy imports in background so first Load click is fast
    import threading

    def _warmup():
        import tensorstore  # noqa: F401
        import scipy.ndimage  # noqa: F401
        try:
            import bioio  # noqa: F401
        except ImportError:
            pass
        # Pre-load registry so discovered datasets are ready
        Registry()

    threading.Thread(target=_warmup, daemon=True).start()

    try:
        tornado.ioloop.IOLoop.current().start()
    except KeyboardInterrupt:
        print("\nStopping...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trailhead web explorer")
    parser.add_argument("--port", type=int, default=9000, help="App server port")
    parser.add_argument("--ng-port", type=int, default=9001, help="Neuroglancer port")
    args = parser.parse_args()
    serve(args.port, args.ng_port)
