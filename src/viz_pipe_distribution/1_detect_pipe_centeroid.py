#!/usr/bin/env python3
"""
1_detect_pipe_centeroid.py

Detect a silver pipe inside black foam using HSV-based foam segmentation,
quadrilateral fitting, Otsu binarization, morphology, and connected component selection.

Command line usage:
    python 1_detect_pipe_centeroid.py --dataset-dir DATASET_DIR [--output-dir OUTPUT_DIR]
                                   [--episode-index N] [--only-aggregate]

Arguments:
    --dataset-dir     Root directory of the dataset (must contain meta/ and videos/)
    --output-dir      Output directory for results (default: ./silver_pipe_circle)
    --episode-index   If provided, only process a single episode index
    --only-aggregate  Skip detection and only generate thumbnail pages and centroid jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
from tqdm import trange


# =========================
# Dataset helper functions
# =========================

def load_meta(dataset_dir: Path) -> dict:
    """Load dataset metadata from meta/info.json."""
    return json.loads((dataset_dir / "meta" / "info.json").read_text())


def resolve_video_path(dataset_dir: Path, episode_index: int, video_key: str) -> Path:
    """Resolve the RGB video path for a given episode index and camera key."""
    meta = load_meta(dataset_dir)
    chunk = episode_index // meta["chunks_size"]
    ep = episode_index % meta["chunks_size"]
    base = dataset_dir / "videos" / f"chunk-{chunk:03d}" / video_key
    for ext in (".mp4", ".avi", ".mkv"):
        candidate = base / f"episode_{ep:06d}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(str(base))


def read_first_rgb_frame(video_path: Path) -> np.ndarray:
    """Read the first frame of a video and return it as an RGB image."""
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# =========================
# ROI extraction
# =========================

def extract_roi(img: np.ndarray, w_range: Tuple[float, float] = (0.0, 0.6), h_range: Tuple[float, float] = (0.0, 0.6)
                ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Extract a rectangular ROI from an image given normalized width and height ranges."""
    img_h, img_w = img.shape[:2]
    rw = int(img_w * (w_range[1] - w_range[0]))
    rh = int(img_h * (h_range[1] - h_range[0]))
    rx = int(img_w * w_range[0])
    ry = int(img_h * h_range[0])
    roi = img[ry : ry + rh, rx : rx + rw].copy()
    return roi, (rx, ry, rw, rh)


# =========================
# Foam segmentation
# =========================

def black_foam_mask_hsv(roi: np.ndarray) -> np.ndarray:
    """Segment black foam in an ROI using HSV thresholds and morphology."""
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    _, s, v = cv2.split(hsv)
    mask = ((v < 120) & (s < 80)).astype(np.uint8) * 255
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def largest_connected_component(mask: np.ndarray, require_bottom: bool = False) -> Optional[np.ndarray]:
    """Return the largest connected component; optionally require it to touch bottom 20%."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if num <= 1:
        return None
    img_h = mask.shape[0]
    bottom_y = int(img_h * 0.8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    for idx in np.argsort(-areas) + 1:
        if require_bottom:
            ys = np.where(labels == idx)[0]
            if not np.any(ys >= bottom_y):
                continue
        return (labels == idx).astype(np.uint8) * 255
    return None


# =========================
# Quadrilateral fitting
# =========================

def order_quad(points: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    s = points.sum(axis=1)
    d = np.diff(points, axis=1).ravel()
    ordered = np.zeros((4, 2), dtype=np.int32)
    ordered[0] = points[np.argmin(s)]
    ordered[2] = points[np.argmax(s)]
    ordered[1] = points[np.argmin(d)]
    ordered[3] = points[np.argmax(d)]
    return ordered


def fit_quadrilateral(mask: np.ndarray, bottom_thresh: float = 0.9) -> Optional[np.ndarray]:
    """Fit a quadrilateral to a binary mask, preferring shapes touching the bottom."""
    img_h = mask.shape[0]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    for k in np.linspace(0.005, 0.05, 20):
        approx = cv2.approxPolyDP(hull, k * peri, True)
        if len(approx) == 4:
            quad = order_quad(approx.reshape(-1, 2))
            if sum(y >= img_h * bottom_thresh for y in quad[:, 1]) >= 2:
                return quad
    rect = cv2.minAreaRect(hull)
    quad = cv2.boxPoints(rect)
    return order_quad(quad.astype(int))


def quad_mask(shape: Tuple[int, int], quad: np.ndarray) -> np.ndarray:
    """Create a filled binary mask from a quadrilateral."""
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [quad], 255)
    return mask


# =========================
# Otsu binarization
# =========================

def otsu_inside(gray: np.ndarray, region_mask: np.ndarray) -> np.ndarray:
    """Apply Otsu thresholding only inside a given region mask."""
    pixels = gray[region_mask > 0]
    if pixels.size == 0:
        return np.zeros_like(gray)
    thresh, _ = cv2.threshold(pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = np.zeros_like(gray)
    binary[(gray >= thresh) & (region_mask > 0)] = 255
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.dilate(binary, kernel, 1)
    binary = cv2.erode(binary, kernel, 1)
    return binary


# =========================
# Episode processing
# =========================

def process_episode(dataset_dir: Path, episode_index: int, output_dir: Path) -> None:
    """Run the full detection pipeline for a single episode index."""
    ep_dir = output_dir / str(episode_index)
    ep_dir.mkdir(parents=True, exist_ok=True)

    video_path = resolve_video_path(dataset_dir, episode_index, "observation.images.chest_rgb")
    rgb = read_first_rgb_frame(video_path)
    cv2.imwrite(str(ep_dir / "0_original.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    roi, (rx, ry, rw, rh) = extract_roi(rgb, (0.3, 0.9), (0.3, 1.0))
    roi_area = rw * rh
    (ep_dir / "roi_meta.json").write_text(json.dumps({"rx": rx, "ry": ry, "rw": rw, "rh": rh}))

    foam = largest_connected_component(black_foam_mask_hsv(roi), require_bottom=True)
    if foam is None:
        print(f"No foam detected in episode {episode_index}")
        return
    cv2.imwrite(str(ep_dir / "1_foam_mask.jpg"), foam)

    foam_rgb = roi.copy()
    foam_rgb[foam == 0] = 0
    cv2.imwrite(str(ep_dir / "2_foam_before_quad.jpg"), cv2.cvtColor(foam_rgb, cv2.COLOR_RGB2BGR))

    quad = fit_quadrilateral(foam)
    if quad is None:
        print(f"Quad fit failed in episode {episode_index}")
        return
    qmask = quad_mask(foam.shape, quad)

    quad_rgb = roi.copy()
    quad_rgb[qmask == 0] = 0
    cv2.imwrite(str(ep_dir / "3_foam_after_quad.jpg"), cv2.cvtColor(quad_rgb, cv2.COLOR_RGB2BGR))

    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    binary = otsu_inside(gray, qmask)
    cv2.imwrite(str(ep_dir / "4_otsu_binary.jpg"), binary)

    dilated = cv2.dilate(binary, np.ones((15, 15), np.uint8), 1)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, 8)

    valid = [i for i in range(1, num) if 0.02 * roi_area <= stats[i, cv2.CC_STAT_AREA] <= 0.2 * roi_area]
    if not valid:
        valid = list(range(1, num))

    img_h = dilated.shape[0]
    selected: List[int] = []
    for i in valid:
        mask_i = (labels == i).astype(np.uint8)
        cnts, _ = cv2.findContours(mask_i, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        has_hole = any(50 < cv2.contourArea(c) < stats[i, cv2.CC_STAT_AREA] for c in cnts)
        bottom_y = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]
        if has_hole or bottom_y >= img_h * 0.95:
            selected.append(i)

    if len(selected) != 1:
        lefts = [stats[i, cv2.CC_STAT_LEFT] for i in valid]
        selected_idx = valid[int(np.argmin(lefts))]
    else:
        selected_idx = selected[0]

    selected_mask = (labels == selected_idx).astype(np.uint8) * 255
    cv2.imwrite(str(ep_dir / "6_selected_cc.jpg"), selected_mask)

    m = cv2.moments(selected_mask)
    if m["m00"] != 0:
        cx_roi = int(m["m10"] / m["m00"])
        cy_roi = int(m["m01"] / m["m00"])
        cx, cy = cx_roi + rx, cy_roi + ry
    else:
        cx, cy = None, None

    (ep_dir / "centroid.json").write_text(json.dumps({"cX": cx, "cY": cy, "valid": True}))

    overlay = rgb.copy()
    roi_view = overlay[ry : ry + rh, rx : rx + rw]
    mask_area = selected_mask > 0
    roi_view[mask_area] = (roi_view[mask_area] * 0.6 + np.array([255, 0, 0]) * 0.4).astype(np.uint8)
    cnts, _ = cv2.findContours(selected_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(roi_view, cnts, -1, (0, 255, 0), 2)
    cv2.imwrite(str(ep_dir / "7_selected_cc_final_overlay.jpg"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


# =========================
# Thumbnail aggregation
# =========================

def generate_thumbnails_all(output_dir: Path, grid_size: int = 4) -> None:
    """Generate grid thumbnails and aggregate all centroids into a jsonl file."""
    ep_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir() and d.name.isdigit()],
                     key=lambda x: int(x.name))
    if not ep_dirs:
        return

    thumbs_nc = output_dir / "thumbnails_no_centroid"
    thumbs_wc = output_dir / "thumbnails_with_centroid"
    thumbs_nc.mkdir(exist_ok=True)
    thumbs_wc.mkdir(exist_ok=True)

    images_per_page = grid_size * grid_size
    pages = (len(ep_dirs) + images_per_page - 1) // images_per_page

    with (output_dir / "all_centroids.jsonl").open("w") as f_agg:
        for page in trange(pages):
            batch = ep_dirs[page * images_per_page : (page + 1) * images_per_page]
            base_img = cv2.imread(str(batch[0] / "7_selected_cc_final_overlay.jpg"))
            meta = json.loads((batch[0] / "roi_meta.json").read_text())
            rx, ry, rw, rh = meta["rx"], meta["ry"], meta["rw"], meta["rh"]
            th, tw = rh, rw

            grids = [np.zeros((th * grid_size, tw * grid_size, 3), np.uint8) for _ in range(2)]
            for i, ep in enumerate(batch):
                r, c = divmod(i, grid_size)
                img = cv2.imread(str(ep / "7_selected_cc_final_overlay.jpg"))[ry:ry + rh, rx:rx + rw]
                grids[0][r * th : (r + 1) * th, c * tw : (c + 1) * tw] = img
                img_c = img.copy()
                data = json.loads((ep / "centroid.json").read_text())
                if data["cX"] is not None:
                    cv2.circle(img_c, (data["cX"] - rx, data["cY"] - ry), 5, (0, 0, 255), -1)
                    f_agg.write(json.dumps({"episode_index": int(ep.name), "cX": data["cX"],
                                            "cY": data["cY"], "valid": data["valid"]}) + "\n")
                if not data.get("valid", True):
                    cv2.putText(img_c, "False", (100, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                (0, 0, 255), 3, cv2.LINE_AA)
                cv2.putText(img_c, f"{ep.name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
                grids[1][r * th : (r + 1) * th, c * tw : (c + 1) * tw] = img_c

            cv2.imwrite(str(thumbs_nc / f"thumbnail_page_{page + 1}.jpg"), grids[0])
            cv2.imwrite(str(thumbs_wc / f"thumbnail_page_{page + 1}.jpg"), grids[1])


# =========================
# Main entry
# =========================

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("./silver_pipe_circle"))
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument("--only-aggregate", action="store_true")
    args = parser.parse_args()

    if args.only_aggregate:
        generate_thumbnails_all(args.output_dir, grid_size=5)
        return

    if args.episode_index is not None:
        process_episode(args.dataset_dir, args.episode_index, args.output_dir)
    else:
        meta = load_meta(args.dataset_dir)
        for idx in trange(meta["total_episodes"]):
            process_episode(args.dataset_dir, idx, args.output_dir)
        generate_thumbnails_all(args.output_dir, grid_size=5)


if __name__ == "__main__":
    main()
