"""Local landmark-guided region masks and image-supported jaw tracing."""
from pathlib import Path

import cv2
import numpy as np
from line_geometry import resample

MODEL = Path(__file__).parent / 'models' / 'face_landmarker.task'
FEATURES = {10, 11, 12, 13, 14, 15, 16}


def detect_regions(rgb, model=MODEL):
    """No photo upload or model download. Return None if no face is detected."""
    import mediapipe as mp
    from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections as Connections
    if not Path(model).is_file():
        raise ValueError('Landmark model missing; run: python setup_models.py (or use --regions off)')
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_buffer=Path(model).read_bytes()),
        num_faces=1, min_face_detection_confidence=0.6,
        min_face_presence_confidence=0.6,
    )
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.face_landmarks:
        return None
    height, width = rgb.shape[:2]
    points = np.array([(p.y * height, p.x * width) for p in result.face_landmarks[0]])
    def indices(name):
        return sorted({i for c in getattr(Connections, name) for i in (c.start, c.end)})
    oval = indices('FACE_LANDMARKS_FACE_OVAL')
    face_width = float(np.ptp(points[oval, 1]))
    pad = max(2, round(face_width * 0.022))
    labels = np.ones((height, width), dtype=np.uint8)  # hair / upper background
    chin_y = points[152, 0]
    labels[np.indices(labels.shape)[0] > chin_y + face_width * 0.08] = 2  # clothes / neck
    face_mask = np.zeros_like(labels)
    cv2.fillConvexPoly(face_mask, cv2.convexHull(points[oval, ::-1].astype(np.int32)), 1)
    labels[face_mask > 0] = 4  # face interior, chiefly shading
    jaw_ids = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152,
               377, 400, 378, 379, 365, 397, 288, 361, 323, 454]
    jaw = points[jaw_ids]
    jaw_band = np.zeros_like(labels)
    cv2.polylines(jaw_band, [jaw[:, ::-1].astype(np.int32)], False, 1, max(5, pad * 4))
    labels[jaw_band > 0] = 3
    groups = [indices('FACE_LANDMARKS_LEFT_EYE'), indices('FACE_LANDMARKS_RIGHT_EYE'),
              indices('FACE_LANDMARKS_LEFT_EYEBROW'), indices('FACE_LANDMARKS_RIGHT_EYEBROW'),
              indices('FACE_LANDMARKS_LIPS'), [1, 2, 98, 327, 129, 358, 94, 49, 279]]
    for label, ids in enumerate(groups, 10):
        mask = np.zeros_like(labels)
        cv2.fillConvexPoly(mask, cv2.convexHull(points[ids, ::-1].astype(np.int32)), 1)
        mask = cv2.dilate(mask, np.ones((pad * 2 + 1, pad * 2 + 1), np.uint8))
        labels[mask > 0] = label
    # The face oval includes the forehead, which can be covered by a fringe.
    brow_y = min(points[groups[2], 0].min(), points[groups[3], 0].min())
    forehead = (np.indices(labels.shape)[0] < brow_y - pad) & (labels == 4)
    labels[forehead] = 1
    return {'labels': labels, 'points': points, 'jaw': jaw, 'face_width': face_width}


def region_of(chain, labels):
    if labels is None:
        return 0
    points = np.rint(np.asarray(chain)).astype(int)
    y = np.clip(points[:, 0], 0, labels.shape[0] - 1)
    x = np.clip(points[:, 1], 0, labels.shape[1] - 1)
    return int(np.bincount(labels[y, x]).argmax())


def glasses_evidence(gray, regions):
    """Conservative closed-ring evidence around eyes; never invent missing rims."""
    if regions is None:
        return False
    points, fw = regions['points'], regions['face_width']
    ink = (gray < 180).astype(np.uint8) * 255
    contours, _ = cv2.findContours(ink, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    eyes = [(points[33] + points[133]) / 2, (points[263] + points[362]) / 2]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if not (fw*.20 < w < fw*.65 and fw*.10 < h < fw*.40):
            continue
        if any(x < eye[1] < x+w and y < eye[0] < y+h for eye in eyes):
            return True
    return False


def protect_glasses(labels, regions):
    """Protect a connected eye/bridge band without replacing original ink."""
    labels = labels.copy()
    p, fw = regions['points'], regions['face_width']
    eyes = p[[33, 133, 263, 362]]
    y0 = max(0, int(eyes[:, 0].min()-fw*.15))
    y1 = min(labels.shape[0], int(eyes[:, 0].max()+fw*.15)+1)
    x0 = max(0, int(eyes[:, 1].min()-fw*.12))
    x1 = min(labels.shape[1], int(eyes[:, 1].max()+fw*.12)+1)
    band = labels[y0:y1, x0:x1]
    band[~np.isin(band, list(FEATURES))] = 16
    return labels


def supported_jaw(regions, gray):
    """Refine the model's jaw in a narrow normal band using visible image gradients."""
    guide = resample(regions['jaw'], 2)
    blurred = cv2.GaussianBlur(gray, (0, 0), 1.5)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    radius = max(2., min(6., regions['face_width'] * 0.018))
    tangents = np.gradient(guide, axis=0)
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-8)
    offsets = np.linspace(-radius, radius, 13)
    candidates = guide[:, None, :] + normals[:, None, :] * offsets[None, :, None]
    y = np.clip(np.rint(candidates[:, :, 0]).astype(int), 0, gray.shape[0] - 1)
    x = np.clip(np.rint(candidates[:, :, 1]).astype(int), 0, gray.shape[1] - 1)
    values = magnitude[y, x]
    scores = values - np.abs(offsets)[None, :] * 1.7
    chosen = np.argmax(scores, axis=1)
    shifts = offsets[chosen]
    evidence = values[np.arange(len(guide)), chosen] >= 5
    # Do not synthesize a whole silhouette from unsupported landmark predictions.
    if np.mean(evidence) < 0.7:
        return None
    longest, current = 0, 0
    for supported in evidence:
        current = 0 if supported else current + 1
        longest = max(longest, current)
    if longest * 2 > regions['face_width'] * 0.12:
        return None
    shifts[~evidence] = 0
    shifts = cv2.GaussianBlur(shifts.astype(np.float32)[:, None], (1, 9), 2).ravel()
    refined = guide + normals * shifts[:, None]
    return [tuple(p) for p in refined]


def region_parameters(label):
    """Multipliers for minimum length and spline tolerance."""
    if label in FEATURES:
        return (0.28 if label in (10, 11) else 0.50), 0.38
    if label == 3:
        return 0.65, 0.8
    if label == 4:
        return 1.4, 0.6
    if label == 2:
        return 2.0, 1.15
    return 1.6, 1.15
