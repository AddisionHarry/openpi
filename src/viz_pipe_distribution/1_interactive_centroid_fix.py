#!/usr/bin/env python3
"""
1_interactive_centroid_fix.py

Interactive tool to manually inspect and correct detected centroids.
Allows mouse clicking to update centroid positions and keyboard shortcuts
to mark validity and navigate between episodes.

Command line usage:
    python 1_interactive_centroid_fix.py --results-dir RESULTS_DIR
        [--window-width 1200] [--window-height 800]

Arguments:
    --results-dir     Directory containing per-episode detection results
    --window-width   Width of the display window in pixels
    --window-height  Height of the display window in pixels
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Tuple, Optional

import cv2
import numpy as np


# =========================
# Argument parsing
# =========================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True, help="Path to detection results directory")
    parser.add_argument("--window-width", type=int, default=1200, help="Display window width")
    parser.add_argument("--window-height", type=int, default=800, help="Display window height")
    return parser.parse_args()


# =========================
# Mouse callback
# =========================

def mouse_callback(event: int, x: int, y: int, flags: int, state: Dict[str, Optional[Tuple[int, int]]]) -> None:
    """Record left mouse click position in shared state."""
    if event == cv2.EVENT_LBUTTONDOWN:
        state["click_pos"] = (x, y)


# =========================
# Utility helpers
# =========================

def ensure_overlay_and_centroid(ep_dir: Path) -> dict:
    """Ensure overlay image and centroid.json exist; create defaults if missing."""
    overlay_path = ep_dir / "7_selected_cc_final_overlay.jpg"
    original_path = ep_dir / "0_original.jpg"
    centroid_path = ep_dir / "centroid.json"

    if not overlay_path.exists():
        if original_path.exists():
            shutil.copy(original_path, overlay_path)
            print(f"Episode {ep_dir.name}: overlay missing, copied from original image")
        else:
            blank = 255 * np.ones((480, 640, 3), np.uint8)
            cv2.imwrite(str(overlay_path), blank)
            print(f"Episode {ep_dir.name}: overlay and original missing, created blank image")

    if not centroid_path.exists():
        centroid_data = {"cX": 0, "cY": 0, "valid": False}
        centroid_path.write_text(json.dumps(centroid_data, indent=2))
        print(f"Episode {ep_dir.name}: centroid.json missing, created default file")
    else:
        centroid_data = json.loads(centroid_path.read_text())

    return centroid_data


def load_existing_modifications(log_path: Path) -> Dict[int, dict]:
    """Load existing modification log if present."""
    mods: Dict[int, dict] = {}
    if not log_path.exists():
        return mods
    with open(log_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            mods[entry["episode_index"]] = {"before": entry["before"], "after": entry["after"]}
    return mods


# =========================
# Main interactive loop
# =========================

def main() -> None:
    """Main interactive centroid editing loop."""
    args = parse_args()
    results_dir = args.results_dir
    window_width = args.window_width
    window_height = args.window_height

    ep_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.isdigit()],
                     key=lambda x: int(x.name))
    if not ep_dirs:
        print("No episode directories found.")
        return

    window_name = "Centroid Editor"
    current_ep_idx = 0
    history = []
    state = {"click_pos": None}

    log_path = results_dir / "modification_log.jsonl"
    episode_mods = load_existing_modifications(log_path)

    while 0 <= current_ep_idx < len(ep_dirs):
        ep_dir = ep_dirs[current_ep_idx]
        centroid_path = ep_dir / "centroid.json"
        centroid_data = ensure_overlay_and_centroid(ep_dir)

        ep_index_int = int(ep_dir.name)
        if ep_index_int not in episode_mods:
            episode_mods[ep_index_int] = {"before": centroid_data.copy(), "after": centroid_data.copy()}

        img = cv2.imread(str(ep_dir / "7_selected_cc_final_overlay.jpg"))
        state["click_pos"] = None

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, window_width, window_height)
        cv2.setMouseCallback(window_name, mouse_callback, state)

        while True:
            h, w = img.shape[:2]
            scale = min(window_width / w, window_height / h)
            disp_w, disp_h = int(w * scale), int(h * scale)
            display_img = cv2.resize(img.copy(), (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)

            if centroid_data.get("cX") is not None and centroid_data.get("cY") is not None:
                cx_disp = int(centroid_data["cX"] * scale)
                cy_disp = int(centroid_data["cY"] * scale)
                cv2.circle(display_img, (cx_disp, cy_disp), 10, (0, 0, 255), -1)

            text = f"Episode: {ep_dir.name}  Valid: {centroid_data.get('valid', True)}"
            color = (255, 255, 255) if centroid_data.get("valid", True) else (0, 0, 255)
            cv2.putText(display_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)

            cv2.imshow(window_name, display_img)
            key = cv2.waitKey(50) & 0xFF

            if state["click_pos"] is not None:
                centroid_data["cX"] = int(state["click_pos"][0] / scale)
                centroid_data["cY"] = int(state["click_pos"][1] / scale)
                state["click_pos"] = None

            episode_mods[ep_index_int]["after"] = centroid_data.copy()

            if key == ord("n"):
                centroid_path.write_text(json.dumps(centroid_data, indent=2))
                history.append(current_ep_idx)
                current_ep_idx += 1
                break
            if key == ord("p"):
                centroid_path.write_text(json.dumps(centroid_data, indent=2))
                current_ep_idx = history.pop() if history else max(0, current_ep_idx - 1)
                break
            if key == ord("f"):
                centroid_data["valid"] = False
            if key == ord("t"):
                centroid_data["valid"] = True
            if key == 27:
                centroid_path.write_text(json.dumps(centroid_data, indent=2))
                current_ep_idx = len(ep_dirs)
                break

    cv2.destroyAllWindows()

    with open(log_path, "w") as f:
        for ep_index in sorted(episode_mods.keys()):
            f.write(json.dumps({
                "episode_index": ep_index,
                "before": episode_mods[ep_index]["before"],
                "after": episode_mods[ep_index]["after"],
            }) + "\n")

    print(f"Modification log saved to {log_path}")


if __name__ == "__main__":
    main()
