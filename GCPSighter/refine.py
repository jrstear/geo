#!/usr/bin/env python3
"""
refine — Color-based pixel refinement for GCP marker detection.

Implements Stage 3 of the emlid2gcp pipeline (geo-56c):
  3a — Anomaly-relative color detection (LAB space); HSV-mask fallback.
  3b — Nearest connected-component search within max_radius.
  3c — Connected-component capture + sanity checks (size, aspect ratio).
  3d — PCA arm-clustering + cv2.fitLine intersection for sub-pixel centre.

Post-passes (operate on the full set of results after per-image refinement):
  R4 — Color consensus: reject detections whose chrominance (a*, b*) deviates
       significantly from the GCP's median color across all images.
  R5 — Bbox consistency: reject detections whose bbox size deviates
       significantly from the GCP's median bbox size.

Target-specific: tuned for orange clay pigeons / spray-painted X marks.
Other target types may need different detection strategies.

Requires: opencv-python, numpy.
"""

import math
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FULL_FRAME_DIAG_MM = math.sqrt(36**2 + 24**2)   # 43.267 mm (duplicated from emlid2gcp)

_MAX_REFINE_RADIUS = 300   # hard cap: give up if no marker found beyond this radius (px)
_SEED_DIST_PENALTY = 0.015 # R1: score /= (1 + k*d) — penalise components far from seed
_MIN_AXIS_RATIO    = 2/3   # λ_min/λ_max threshold; components below this are too
                           # elongated to be an X marker (lane lines score ~0.01–0.05).
                           # Asymmetric or partially-occluded real markers may need
                           # this lowered to ~0.3–0.4.
_MARKER_SIZE_M     = 0.5   # R3: expected physical marker size (metres).  Calibrated from
                           # 46 good-case confirmed pairs: median 0.55 m, range 0.17–0.81 m.
                           # Used with GSD to compute expected bbox size in pixels.
_MIN_MARKER_PX     = 10    # R3: hard floor — reject any component with bbox avg < this,
                           # regardless of GSD.  No real marker is this small.

# R4: color consensus post-pass — DISABLED.  Median-based chrominance consensus
# fails when (a) bad detections form the majority (median reflects wrong feature)
# or (b) correct detections have varied chrominance from viewing-angle/exposure
# while wrong detections cluster near the median.  Tested on ghostrider gulch:
# R4 caused 14 false rejections of good detections (+9 bad).  See
# diagnose_postpass.py for per-GCP color analysis.  Keep the code and constant
# in case a better consensus method (score-weighted, DBSCAN clustering) is tried.
_COLOR_CONSENSUS_THRESH = 15.0

# R5: bbox consistency post-pass — reject detections whose bbox avg size
# deviates from the GCP's median by more than this ratio.
# Calibrated from 85 confirmed pairs: good cases stay within 0.71-1.39 ratio;
# bad cases go to 0.25 or 5.91.  Threshold 1.5 (accept 0.67-1.50) catches
# wrong-feature detections with 0 false rejections.  Tested 2.5→2.0→1.5→1.4:
# 1.5 is optimal (12 improved, 0 worsened vs R3 baseline).
_BBOX_CONSISTENCY_THRESH = 1.5

# R4/R5: minimum detections per GCP to compute consensus.
# With only 1 detection, there's no consensus to compare against.
_MIN_CONSENSUS_SAMPLES = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_gsd(exif: dict) -> Optional[float]:
    """Compute GSD in metres/pixel from EXIF metadata."""
    focal_mm = exif.get('focal_mm')
    focal35  = exif.get('focal35_mm')
    img_w    = exif.get('img_w')
    rel_alt  = exif.get('rel_alt')
    if not all([focal_mm, focal35, img_w, rel_alt]):
        return None
    if focal_mm <= 0 or focal35 <= 0 or rel_alt <= 1.0:
        return None
    scale = focal_mm / focal35
    sensor_diag = FULL_FRAME_DIAG_MM * scale
    img_h = exif.get('img_h', img_w)
    aspect = img_w / img_h
    sensor_h = sensor_diag / math.sqrt(1 + aspect ** 2)
    sensor_w = sensor_h * aspect
    return (rel_alt * sensor_w) / (focal_mm * img_w)


# ---------------------------------------------------------------------------
# Stage 3 — Pixel Refinement (geo-56c)
# ---------------------------------------------------------------------------

def _refine_single(image_path: str, px: float, py: float,
                   max_radius: int = _MAX_REFINE_RADIUS,
                   gsd_m: Optional[float] = None) -> Optional[dict]:
    """
    Refine a single (image, GCP) pixel estimate using color analysis.

    Implements the four-stage pipeline from docs/gcp_object_identification.md §3:
      3a — Anomaly-relative color detection (LAB space); HSV-mask fallback.
      3b — Nearest connected-component search within max_radius (expanding-annuli
           equivalent: select the component whose closest pixel is nearest the seed).
      3c — Connected-component capture + sanity checks (size, aspect ratio).
      3d — PCA arm-clustering + cv2.fitLine intersection for sub-pixel centre.

    Returns a dict {px, py, confidence, marker_bbox} on success, or None on failure.
    The caller retains the original estimate on failure.

    Requires: opencv-python, numpy.
    """
    try:
        import cv2          # type: ignore
        import numpy as np
    except ImportError:
        return None

    img = cv2.imread(image_path)
    if img is None:
        return None
    ih, iw = img.shape[:2]
    px_i = int(round(max(0.0, min(float(iw - 1), px))))
    py_i = int(round(max(0.0, min(float(ih - 1), py))))

    # Crop a working window to avoid analysing the entire image.
    # Extra 110 px beyond max_radius allows background-ring sampling.
    pad  = max_radius + 110
    cx1  = max(0,  px_i - pad)
    cy1  = max(0,  py_i - pad)
    cx2  = min(iw, px_i + pad)
    cy2  = min(ih, py_i + pad)
    crop = img[cy1:cy2, cx1:cx2]
    ch, cw = crop.shape[:2]

    # Seed position in crop coordinates
    sx = px_i - cx1
    sy = py_i - cy1

    # Pre-compute per-pixel distance from seed
    ys, xs = np.mgrid[0:ch, 0:cw]
    dists = np.sqrt((xs - sx) ** 2 + (ys - sy) ** 2).astype(np.float32)

    # ------------------------------------------------------------------
    # Stage 3a — Anomaly-relative detection (LAB space)
    # Background ring: 55–90 px from seed — far enough to miss a typical
    # GCP marker but within the crop window.
    # crop_lab and deviation_map are always computed here so the scoring
    # loop can use them regardless of which detection path succeeded.
    # ------------------------------------------------------------------
    crop_lab = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab)
    deviation_map: Optional[np.ndarray] = None   # per-pixel max LAB deviation (ch, cw)
    marker_mask: Optional[np.ndarray] = None
    ring_mask = (dists >= 55) & (dists <= 90)
    if int(ring_mask.sum()) >= 50:
        ring_lab = crop_lab[ring_mask].astype(np.float32)
        bg_mean  = ring_lab.mean(axis=0)
        bg_std   = ring_lab.std(axis=0) + 1.0          # avoid divide-by-zero
        lab_flat  = crop_lab.reshape(-1, 3).astype(np.float32)
        dev       = np.abs(lab_flat - bg_mean) / bg_std
        deviation_map = dev.max(axis=1).reshape(ch, cw)
        anomaly   = (deviation_map > 2.5) & (dists <= max_radius)
        if int(anomaly.sum()) >= 15:
            marker_mask = anomaly

    # Stage 3a fallback — per-colour HSV masks (orange → red → white)
    if marker_mask is None or int(marker_mask.sum()) < 15:
        crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        within_r = dists <= max_radius

        def _hsv(lo, hi):
            return cv2.inRange(crop_hsv,
                               np.array(lo, np.uint8),
                               np.array(hi, np.uint8)).astype(bool) & within_r

        for cand in [
            _hsv([5,  120, 120], [25, 255, 255]),                          # orange
            _hsv([0,  120, 120], [5,  255, 255]) |                         # red
            _hsv([170, 120, 120], [180, 255, 255]),
            _hsv([0,   0,  200], [180, 40,  255]),                         # white
        ]:
            if int(cand.sum()) >= 15:
                marker_mask = cand
                break

    if marker_mask is None or int(marker_mask.sum()) < 15:
        return None     # no marker signal detected

    # ------------------------------------------------------------------
    # Stage 3b+3c — Connected components; score all candidates within
    # max_radius and select the best-scoring one.
    # ------------------------------------------------------------------
    num_labels, labels_img = cv2.connectedComponents(
        marker_mask.astype(np.uint8), connectivity=8)

    # Collect all components within max_radius, compute PCA for each, apply
    # sanity checks, and score survivors.  The score
    #
    #   score = ratio × size × mean_dev × coherence × dist_penalty × size_penalty
    #
    # dist_penalty = 1/(1 + k*d): penalises features far from seed (R1).
    # size_penalty = min(r, 1/r) where r = bbox/expected: penalises bboxes
    #   that deviate from the GSD-predicted marker size (R3).  Peaks at 1.0
    #   when bbox matches expected; drops to 0.1 at 10× mismatch.
    #
    max_pixels = (max_radius * 2) ** 2 // 3
    scored: list = []   # (score, ratio, eigvals, eigvecs, r_idx, c_idx, pyx, ctr, cen)

    for lab in range(1, num_labels):
        r_idx, c_idx = np.where(labels_img == lab)
        d = float(np.sqrt((c_idx - sx) ** 2 + (r_idx - sy) ** 2).min())
        if d > max_radius:
            continue

        n = len(r_idx)
        bh = int(r_idx.max() - r_idx.min() + 1)
        bw = int(c_idx.max() - c_idx.min() + 1)
        bbox_avg = (bw + bh) / 2.0
        if bbox_avg < _MIN_MARKER_PX:
            continue   # R3: hard floor — too small to be any real marker
        if n < 20 or n > max_pixels:
            continue
        if max(bh, bw) / max(1, min(bh, bw)) > 15:
            continue
        if n < 6:
            continue   # too few pixels for reliable PCA

        pyx = np.column_stack([r_idx, c_idx]).astype(np.float64)
        ctr = pyx.mean(axis=0)
        cen = pyx - ctr
        eigvals, eigvecs = np.linalg.eigh(np.cov(cen.T))
        ratio = eigvals[0] / eigvals[1] if eigvals[1] > 1e-6 else 0.0
        if ratio < _MIN_AXIS_RATIO:
            continue   # too elongated — not an X shape

        # Idea 1 — mean background deviation: how strongly do the component
        # pixels deviate from local background?  Vivid orange paint scores
        # high; subtle oil spots and shadows score low.
        if deviation_map is not None:
            mean_dev = float(deviation_map[r_idx, c_idx].mean())
        else:
            mean_dev = 1.0   # neutral when background ring unavailable

        # Idea 3 — color coherence: real markers have consistent color across
        # their pixels; mottled ground, oil spots, and mixed-material blobs
        # have scattered LAB chroma (a*, b* channels).
        comp_ab  = crop_lab[r_idx, c_idx].astype(np.float32)[:, 1:]  # a*, b* only
        coherence = 1.0 / (1.0 + float(comp_ab.std()))

        # R1: soft penalty for distance from projection seed — the projection
        # estimate is a strong prior, so nearby features should be preferred.
        # Uses centroid-to-seed distance (not nearest-pixel) for stability.
        centroid_dist = float(np.sqrt((ctr[1] - sx) ** 2 + (ctr[0] - sy) ** 2))
        dist_penalty = 1.0 / (1.0 + _SEED_DIST_PENALTY * centroid_dist)

        # R3: GSD-based size penalty — penalise components whose bbox deviates
        # from the expected marker size.  Calibrated from confirmed data: good
        # cases have ratio 0.31–1.49 vs expected; bad cases go to 0.04 or 4.3.
        # size_penalty peaks at 1.0 when bbox matches expected, drops linearly.
        if gsd_m is not None and gsd_m > 0:
            expected_px = _MARKER_SIZE_M / gsd_m
            size_ratio = bbox_avg / expected_px
            size_penalty = min(size_ratio, 1.0 / size_ratio) if size_ratio > 0 else 0.0
        else:
            size_penalty = 1.0   # neutral when GSD unavailable

        score = ratio * (eigvals[0] + eigvals[1]) * mean_dev * coherence * dist_penalty * size_penalty
        scored.append((score, ratio, eigvals, eigvecs, r_idx, c_idx, pyx, ctr, cen))

    if not scored:
        return None

    scored.sort(key=lambda t: -t[0])   # highest score first
    _, _ratio, _eigvals, best_eigvecs, comp_rows, comp_cols, pix_yx, centroid, centered = scored[0]
    kept_eigvecs = best_eigvecs

    n_pix = len(comp_rows)

    # ------------------------------------------------------------------
    # Stage 3d — PCA arm-clustering → line intersection (sub-pixel centre)
    # ------------------------------------------------------------------
    refined_col = float(centroid[1])   # default: component centroid
    refined_row = float(centroid[0])

    if n_pix >= 6 and kept_eigvecs is not None:
        eigvecs = kept_eigvecs   # reuse computation from shape check above
        # eigvecs[:, 1] is the dominant principal direction (most variance)
        proj1    = centered @ eigvecs[:, 1]
        proj2    = centered @ eigvecs[:, 0]
        arm1_idx = np.abs(proj1) >= np.abs(proj2)
        arm1_pts = pix_yx[arm1_idx]
        arm2_pts = pix_yx[~arm1_idx]

        if len(arm1_pts) >= 3 and len(arm2_pts) >= 3:
            # cv2.fitLine expects (x, y) = (col, row) → swap axes
            pts1 = arm1_pts[:, ::-1].astype(np.float32).reshape(-1, 1, 2)
            pts2 = arm2_pts[:, ::-1].astype(np.float32).reshape(-1, 1, 2)
            l1 = cv2.fitLine(pts1, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            l2 = cv2.fitLine(pts2, cv2.DIST_L2, 0, 0.01, 0.01).flatten()

            vx1, vy1, x0_1, y0_1 = (float(v) for v in l1)
            vx2, vy2, x0_2, y0_2 = (float(v) for v in l2)
            denom = vx1 * vy2 - vy1 * vx2
            if abs(denom) > 1e-6:
                dx = x0_2 - x0_1
                dy = y0_2 - y0_1
                t  = (dx * vy2 - dy * vx2) / denom
                xi = x0_1 + t * vx1
                yi = y0_1 + t * vy1
                # Accept intersection only when it falls inside the component bbox
                if (float(comp_cols.min()) <= xi <= float(comp_cols.max()) and
                        float(comp_rows.min()) <= yi <= float(comp_rows.max())):
                    refined_col = xi
                    refined_row = yi

    # Map refined crop-coords back to full-image pixel coords
    full_px = refined_col + cx1
    full_py = refined_row + cy1

    bbox_x1 = int(comp_cols.min()) + cx1
    bbox_y1 = int(comp_rows.min()) + cy1
    bbox_x2 = int(comp_cols.max()) + cx1
    bbox_y2 = int(comp_rows.max()) + cy1

    # Compute marker median LAB color for R4 post-pass consensus check.
    # Uses chrominance channels (a*, b*) which are lighting-invariant.
    comp_lab = crop_lab[comp_rows, comp_cols].astype(np.float32)
    marker_L = float(np.median(comp_lab[:, 0]))
    marker_a = float(np.median(comp_lab[:, 1]))
    marker_b = float(np.median(comp_lab[:, 2]))

    bw = bbox_x2 - bbox_x1
    bh = bbox_y2 - bbox_y1
    bbox_avg = (bw + bh) / 2.0

    return {
        'px':          full_px,
        'py':          full_py,
        'confidence':  'color_refined',
        'marker_bbox': f'{bbox_x1},{bbox_y1},{bbox_x2},{bbox_y2}',
        'marker_lab':  (marker_L, marker_a, marker_b),
        'bbox_avg':    bbox_avg,
    }


# ---------------------------------------------------------------------------
# Post-pass filters (R4, R5) — operate on the full set after per-image refinement
# ---------------------------------------------------------------------------

def _revert_to_projection(est: dict) -> None:
    """Revert a refined estimate back to its original projection coordinates."""
    if '_orig_px' in est:
        est['px'] = est['_orig_px']
        est['py'] = est['_orig_py']
    est['confidence'] = 'projection'
    est.pop('marker_bbox', None)
    est.pop('_marker_lab', None)
    est.pop('_bbox_avg', None)


def _postpass_color_consensus(estimates: Dict[str, Dict[str, dict]]) -> int:
    """
    R4 — Color consensus post-pass.

    For each GCP, compute the median chrominance (a*, b*) across all refined
    detections.  Reject any detection whose chrominance distance from the
    median exceeds _COLOR_CONSENSUS_THRESH and revert it to the original
    projection estimate.

    Lightness (L*) is excluded because it varies significantly with sun angle,
    cloud cover, and camera exposure.  Chrominance (a*, b*) is much more stable
    for the same colored marker.

    Returns the number of detections reverted.
    """
    import numpy as np
    reverted = 0
    for gcp_label, img_map in estimates.items():
        # Collect all refined detections with marker_lab data
        refined = []
        for img_name, est in img_map.items():
            if est.get('confidence') == 'color_refined' and est.get('_marker_lab'):
                refined.append((img_name, est))
        if len(refined) < _MIN_CONSENSUS_SAMPLES:
            continue

        # Compute median a*, b* (chrominance consensus)
        a_vals = [est['_marker_lab'][1] for _, est in refined]
        b_vals = [est['_marker_lab'][2] for _, est in refined]
        med_a = float(np.median(a_vals))
        med_b = float(np.median(b_vals))

        # Check each detection against consensus
        for img_name, est in refined:
            da = est['_marker_lab'][1] - med_a
            db = est['_marker_lab'][2] - med_b
            ab_dist = math.sqrt(da * da + db * db)
            if ab_dist > _COLOR_CONSENSUS_THRESH:
                _revert_to_projection(est)
                reverted += 1
    return reverted


def _postpass_bbox_consistency(estimates: Dict[str, Dict[str, dict]]) -> int:
    """
    R5 — Bbox size consistency post-pass.

    For each GCP, compute the median bbox avg size across all refined
    detections.  Reject any detection whose size ratio (detection / median)
    exceeds _BBOX_CONSISTENCY_THRESH or falls below 1/threshold, and revert
    it to the original projection estimate.

    Returns the number of detections reverted.
    """
    import numpy as np
    reverted = 0
    for gcp_label, img_map in estimates.items():
        # Collect all refined detections with bbox_avg data
        refined = []
        for img_name, est in img_map.items():
            if est.get('confidence') == 'color_refined' and est.get('_bbox_avg'):
                refined.append((img_name, est))
        if len(refined) < _MIN_CONSENSUS_SAMPLES:
            continue

        # Compute median bbox size
        sizes = [est['_bbox_avg'] for _, est in refined]
        med_size = float(np.median(sizes))
        if med_size < 1.0:
            continue

        # Check each detection against consensus
        for img_name, est in refined:
            ratio = est['_bbox_avg'] / med_size
            if ratio > _BBOX_CONSISTENCY_THRESH or ratio < 1.0 / _BBOX_CONSISTENCY_THRESH:
                _revert_to_projection(est)
                reverted += 1
    return reverted


def refine_all_estimates(
        estimates: Dict[str, Dict[str, dict]],
        exif_map: Dict[str, dict],
        threads: int = 0,
        gcp_order: Optional[List[str]] = None,
        refine_limit: int = 0) -> Dict[str, Dict[str, dict]]:
    """
    Run Stage-3 pixel refinement on (GCP, image) pairs in *estimates*.

    gcp_order:    GCP labels in structural priority order.  When provided, tasks
                  are built in this order so that --refine-limit keeps the most
                  important pairs.  If None, dict insertion order is used.
    refine_limit: if > 0, process at most this many (gcp, image) pairs.
                  SPA-* pairs are excluded before the limit is applied, so the
                  limit budget is spent entirely on GCP-* and CHK-* pairs.

    Estimates that succeed have their px/py replaced with the refined sub-pixel
    coordinate and gain 'confidence' and 'marker_bbox' keys.  Pairs that fail
    refinement retain their original projection values unchanged.

    Returns the mutated *estimates* dict.
    """
    try:
        import cv2      # type: ignore  # noqa: F401
        import numpy    # type: ignore  # noqa: F401
    except ImportError:
        print("  WARNING: opencv-python or numpy not installed — --refine-pixels skipped.")
        return estimates

    label_order = gcp_order if gcp_order else list(estimates.keys())

    tasks: List[Tuple] = []

    for gcp_label in label_order:
        img_map = estimates.get(gcp_label)
        if not img_map:
            continue
        # Skip SPA-* (spare) points — refinement cost is not justified.
        if gcp_label.startswith('SPA-'):
            continue
        for img_name, est in img_map.items():
            exif = exif_map.get(img_name) or {}
            path = exif.get('path')
            if path:
                gsd = _compute_gsd(exif)
                tasks.append((gcp_label, img_name, path, est['px'], est['py'], gsd))

    if refine_limit > 0 and len(tasks) > refine_limit:
        print(f"  Limiting to {refine_limit} of {len(tasks)} (GCP,image) pairs (--refine-limit).")
        tasks = tasks[:refine_limit]

    if not tasks:
        return estimates

    n_threads = threads or cpu_count()
    total     = len(tasks)
    refined   = 0
    print(f"Refining estimates via color analysis using {n_threads} threads...")

    def _worker(task):
        gcp_label, img_name, path, px_seed, py_seed, gsd = task
        return gcp_label, img_name, _refine_single(path, px_seed, py_seed, gsd_m=gsd)

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        for done, (gcp_label, img_name, result) in enumerate(
                executor.map(_worker, tasks), start=1):
            if result is not None:
                est = estimates[gcp_label][img_name]
                # Store original projection estimate so post-passes can revert
                est['_orig_px'] = est['px']
                est['_orig_py'] = est['py']
                est['px']          = result['px']
                est['py']          = result['py']
                est['confidence']  = result['confidence']
                est['marker_bbox'] = result['marker_bbox']
                est['_marker_lab'] = result.get('marker_lab')
                est['_bbox_avg']   = result.get('bbox_avg')
                refined += 1
            if done % max(1, total // 20) == 0 or done == total:
                hit_pct = 100 * refined / done if done else 0
                print(f"  {100 * done / total:5.1f}% done: {done} of {total} GCP,image pairs "
                      f"analyzed, {refined} estimates changed ({hit_pct:.0f}%)",
                      end='\r', flush=True)

    print()

    # Post-passes: operate on the full set of results after per-image refinement.
    # These compare each detection against the GCP's consensus and revert outliers.
    # R4 (color consensus) is disabled: median-based consensus fails when
    # bad detections form the majority or have similar chrominance to correct ones.
    # See diagnose_postpass.py analysis — GCP-112 and GCP-96 show false rejections.
    # r4 = _postpass_color_consensus(estimates)
    r5 = _postpass_bbox_consistency(estimates)
    if r5:
        print(f"  Post-pass (R5 bbox) reverted {r5} detection(s) to projection estimates.")

    return estimates
