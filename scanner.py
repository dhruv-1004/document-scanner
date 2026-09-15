"""
Document Scanner v2 - multi-strategy edge/contour detection.

Improvements over v1:
  - Corrects camera EXIF rotation before processing.
  - Tries several detection strategies (different Canny thresholds,
    adaptive threshold, HSV-saturation, LAB-lightness) instead of one.
  - Scores every candidate quadrilateral by area-coverage * rectangularity
    and keeps the best one found across all strategies.
  - Falls back to a rotated bounding box (minAreaRect) when a contour is
    roughly rectangular but wavy/noisy (e.g. tape, glare) instead of
    giving up outright.
  - Only falls back to the uncropped full image when NO strategy finds
    anything plausible at all.
"""

import argparse
import os
import cv2
import numpy as np

try:
    from PIL import Image, ExifTags
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


def load_image_corrected(path):
    """Load an image and apply EXIF rotation if present."""
    if _HAS_PIL:
        try:
            from PIL import ImageOps
            pil_img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            pass
    return cv2.imread(path)


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    width = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
    height = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))
    width, height = max(width, 1), max(height, 1)
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (width, height))


def rectangularity(contour):
    area = cv2.contourArea(contour)
    if area <= 0:
        return 0.0
    rect = cv2.minAreaRect(contour)
    rect_area = rect[1][0] * rect[1][1]
    if rect_area <= 0:
        return 0.0
    return min(area / rect_area, 1.0)


def edge_map_canny(gray, low, high):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, low, high)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)


def edge_map_adaptive(gray):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    th = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 10)
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)


def edge_map_saturation(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = cv2.GaussianBlur(hsv[:, :, 1], (5, 5), 0)
    _, th = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)


def edge_map_lightness(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = cv2.GaussianBlur(lab[:, :, 0], (5, 5), 0)
    _, th = cv2.threshold(L, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)


def best_quad_from_mask(mask, image_area, min_area_ratio=0.15):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0
    best_pts, best_score = None, 0.0
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:6]:
        area = cv2.contourArea(c)
        if area < min_area_ratio * image_area:
            continue
        rectang = rectangularity(c)
        if rectang < 0.55:
            continue
        peri = cv2.arcLength(c, True)
        pts = None
        for eps_frac in (0.01, 0.02, 0.03, 0.05, 0.08):
            approx = cv2.approxPolyDP(c, eps_frac * peri, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype("float32")
                break
        if pts is None:
            pts = cv2.boxPoints(cv2.minAreaRect(c)).astype("float32")
        score = (area / image_area) * rectang
        if score > best_score:
            best_pts, best_score = pts, score
    return best_pts, best_score


def find_document(image, debug=False, debug_prefix="debug"):
    ratio = image.shape[0] / 500.0
    small = cv2.resize(image, (int(image.shape[1] / ratio), 500))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    image_area = small.shape[0] * small.shape[1]
    strategies = {
        "canny_75_200": lambda: edge_map_canny(gray, 75, 200),
        "canny_30_100": lambda: edge_map_canny(gray, 30, 100),
        "adaptive": lambda: edge_map_adaptive(gray),
        "saturation": lambda: edge_map_saturation(small),
        "lightness": lambda: edge_map_lightness(small),
    }
    best_pts, best_score, best_name = None, 0.0, None
    for name, fn in strategies.items():
        mask = fn()
        if debug:
            cv2.imwrite(f"{debug_prefix}_{name}.jpg", mask)
        pts, score = best_quad_from_mask(mask, image_area)
        if score > best_score:
            best_pts, best_score, best_name = pts, score, name
    if debug:
        print(f"[debug] best strategy: {best_name}  score={best_score:.3f}")
        if best_pts is not None:
            dbg = small.copy()
            cv2.drawContours(dbg, [best_pts.astype(int)], -1, (0, 255, 0), 2)
            cv2.imwrite(f"{debug_prefix}_best_quad.jpg", dbg)
    if best_pts is None or best_score < 0.12:
        return None, best_score, best_name
    return best_pts * ratio, best_score, best_name


def scan(image, debug=False, color=False, debug_prefix="debug"):
    pts, score, strategy = find_document(image, debug=debug, debug_prefix=debug_prefix)
    if pts is None:
        print(f"[warn] No confident document outline found (best score={score:.3f}). Falling back to full frame.")
        warped = image
    else:
        print(f"[ok] Document detected via '{strategy}' strategy (score={score:.3f}).")
        warped = four_point_transform(image, pts)
        if image.shape[0] >= image.shape[1] and warped.shape[1] > warped.shape[0]:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    if color:
        return warped
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    scanned = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 12)
    return cv2.medianBlur(scanned, 3)


def run_image(path, debug=False, color=False):
    image = load_image_corrected(path)
    if image is None:
        raise SystemExit(f"[error] Could not read image: {path}")
    base, ext = os.path.splitext(path)
    prefix = f"debug_{os.path.basename(base)}"
    result = scan(image, debug=debug, color=color, debug_prefix=prefix)
    out_path = f"{base}_scanned_v2{ext}"
    cv2.imwrite(out_path, result)
    print(f"[ok] Input : {path}  ({image.shape[1]}x{image.shape[0]})")
    print(f"[ok] Output: {out_path}  ({result.shape[1]}x{result.shape[0]})")
    return out_path


def run_webcam(debug=False):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("[error] Could not open webcam.")
    print("Press 's' to scan the current frame, 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        preview = frame.copy()
        pts, _, _ = find_document(frame)
        if pts is not None:
            cv2.drawContours(preview, [pts.astype(int)], -1, (0, 255, 0), 2)
        cv2.imshow("Document Scanner v2 - live", preview)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            result = scan(frame, debug=debug)
            cv2.imwrite("webcam_scanned_v2.jpg", result)
            print("[ok] Saved webcam_scanned_v2.jpg")
    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Scan a document from a photo (v2, multi-strategy).")
    ap.add_argument("--image", help="path to the input image")
    ap.add_argument("--webcam", action="store_true")
    ap.add_argument("--debug", action="store_true", help="save per-strategy mask images")
    ap.add_argument("--color", action="store_true", help="skip thresholding, keep colour")
    args = ap.parse_args()
    if args.webcam:
        run_webcam(debug=args.debug)
    elif args.image:
        run_image(args.image, debug=args.debug, color=args.color)
    else:
        ap.error("provide --image <path> or --webcam")


if __name__ == "__main__":
    main()
