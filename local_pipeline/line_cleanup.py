"""Region-aware removal of stray marks without changing portrait geometry."""

import cv2
import numpy as np
from scipy.spatial import cKDTree

from line_geometry import resample
from portrait_regions import FEATURES, region_of

DETAIL_LEVELS = ('faithful', 'clean', 'minimal', 'contour')


def budget_chains(chains, labels, gray, budget=24, threshold=180, foreground_mask=None):
    """Budget optional texture only; facial marks, silhouette and junctions survive."""
    if budget is None:
        return chains, {'removed_by_budget': 0, 'detail_budget': None}
    if budget < 0:
        raise ValueError('Detail budget must be nonnegative')
    scale = max(gray.shape) / 800
    ink = gray < threshold
    valid_rows = ink.any(axis=1)
    left = ink.argmax(axis=1)
    right = ink.shape[1] - 1 - ink[:, ::-1].argmax(axis=1)
    valid_columns = ink.any(axis=0)
    top = ink.argmax(axis=0)
    bottom = ink.shape[0] - 1 - ink[::-1].argmax(axis=0)
    silhouette_distance = None
    if foreground_mask is not None:
        mask = (foreground_mask > 0).astype(np.uint8)
        edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        silhouette_distance = cv2.distanceTransform(1-edge, cv2.DIST_L2, 5)
    endpoint_owners = {}
    for i, chain in enumerate(chains):
        for endpoint in (chain[0], chain[-1]):
            endpoint_owners.setdefault(tuple(endpoint), set()).add(i)
    protected, optional = set(), []
    for i, chain in enumerate(chains):
        ij = np.rint(chain).astype(int)
        y = np.clip(ij[:, 0], 0, gray.shape[0] - 1)
        x = np.clip(ij[:, 1], 0, gray.shape[1] - 1)
        region = region_of(chain, labels)
        boundary = valid_rows[y] & (np.minimum(abs(x-left[y]), abs(x-right[y])) <= max(1., scale))
        boundary |= valid_columns[x] & (np.minimum(abs(y-top[x]), abs(y-bottom[x])) <= max(1., scale))
        if silhouette_distance is not None:
            boundary = silhouette_distance[y, x] <= max(2., 3*scale)
        connected = all(len(endpoint_owners[tuple(p)]) > 1 for p in (chain[0], chain[-1]))
        # An absent face detector cannot safely distinguish detail from identity.
        if labels is None or region in FEATURES or region == 3 or np.mean(boundary) >= .6 or connected:
            protected.add(i)
            continue
        length = np.linalg.norm(np.diff(np.asarray(chain), axis=0), axis=1).sum() / scale
        evidence = np.mean(1 - gray[y, x].astype(float) / 255)
        score = np.log1p(length) * (0.5 + evidence)
        optional.append((score, i))
    retained = protected | {i for _, i in sorted(optional, reverse=True)[:budget]}
    return [c for i, c in enumerate(chains) if i in retained], {
        'detail_budget': budget, 'protected_chains': len(protected),
        'optional_chains': len(optional), 'removed_by_budget': len(chains)-len(retained)}


def clean_feature_ink(ink, labels, detail):
    """Clean each eye/brow separately; keep meaningful eye holes and thin marks."""
    result = np.zeros_like(ink)
    scale = max(ink.shape) / 800
    radius = max(1, round(scale))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    for label in (10, 11, 12, 13):
        region = labels == label
        mask = np.where(region, ink, 0).astype(np.uint8)
        if detail != 'faithful':
            if label >= 12:
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            sigma = max(.3, scale * (1. if label >= 12 else .3))
            mask = (cv2.GaussianBlur(mask, (0, 0), sigma) >= 128).astype(np.uint8) * 255
            mask[~region] = 0
            count, components, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            cutoff = max(2, round((4 if label >= 12 else 2) * scale**2))
            if label >= 12 and count > 1:
                cutoff = max(cutoff, stats[1:, cv2.CC_STAT_AREA].max() * .06)
            for index in range(1, count):
                if stats[index, cv2.CC_STAT_AREA] < cutoff:
                    mask[components == index] = 0
            if label >= 12:
                # Brow speckles become coherent silhouettes; eyes keep their holes.
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(mask, contours, -1, 255, cv2.FILLED)
                mask[~region] = 0
        result = cv2.bitwise_or(result, mask)
    return result


def select_clean_chains(chains, labels, gray, min_length, detail, threshold=180):
    """Prefer long supported strokes; protect facial marks and structural connectors."""
    if detail not in DETAIL_LEVELS:
        raise ValueError(f'Unknown detail level: {detail}')
    scale = max(gray.shape) / 800
    lengths = np.array([cv2.arcLength(np.asarray(c, np.float32), False) for c in chains])
    regions = [region_of(c, labels) for c in chains]
    baseline = np.array([max(2, min_length * .2) if r in FEATURES else min_length for r in regions])
    keep = lengths >= baseline
    if detail == 'faithful' or not chains:
        return [c for c, yes in zip(chains, keep) if yes], {'removed_strays': 0, 'removed_nearby': 0}

    strength = 1.45 if detail == 'minimal' else 1.
    # Hair and clothing labels are approximate landmark regions, not segmentation.
    cutoffs = {0: 28, 1: 34, 2: 38, 3: 12, 4: 24}
    targets = np.array([max(b, cutoffs.get(r, 0) * scale * strength)
                        if r not in FEATURES else b for r, b in zip(regions, baseline)])
    keep &= lengths >= targets
    # Retain measured silhouette fragments, even when shorter than the texture cutoff.
    ink = gray < threshold
    rows_with_ink = ink.any(axis=1)
    left = ink.argmax(axis=1)
    right = ink.shape[1] - 1 - ink[:, ::-1].argmax(axis=1)
    for i, chain in enumerate(chains):
        if keep[i] or lengths[i] < baseline[i] or regions[i] in FEATURES:
            continue
        ij = np.rint(chain).astype(int)
        y = np.clip(ij[:, 0], 0, gray.shape[0] - 1)
        x = np.clip(ij[:, 1], 0, gray.shape[1] - 1)
        boundary = rows_with_ink[y] & (np.minimum(abs(x - left[y]), abs(x - right[y])) <= max(1., scale))
        if np.mean(boundary) >= .6:
            keep[i] = True
    samples = [resample(c) for c in chains]
    coordinates = np.concatenate(samples)
    owners = np.repeat(np.arange(len(chains)), [len(c) for c in samples])
    tree = cKDTree(coordinates)
    # A short connector between two retained strokes is structure, not a stray.
    initial_keep = keep.copy()
    for i, chain in enumerate(chains):
        if keep[i] or lengths[i] < baseline[i]:
            continue
        neighbors = tree.query_ball_point(np.asarray([chain[0], chain[-1]]), max(1., scale))
        anchors = [{int(owners[j]) for j in group if owners[j] != i and initial_keep[owners[j]]}
                   for group in neighbors]
        if anchors[0] and anchors[1] and len(anchors[0] | anchors[1]) >= 2:
            keep[i] = True
    removed_strays = int(np.count_nonzero((lengths >= baseline) & ~keep))

    accepted = []
    removed_nearby = 0
    # Compare only same-region non-facial strokes, longest first.
    for i in sorted(np.flatnonzero(keep), key=lambda j: -lengths[j]):
        if regions[i] in FEATURES or regions[i] in (3, 4):
            continue
        points = resample(chains[i])
        lo, hi = points.min(axis=0), points.max(axis=0)
        radius = (3. if detail == 'minimal' else 2.3) * scale
        duplicate = False
        for region, other_lo, other_hi, other_tree in accepted:
            if region != regions[i] or np.any(hi + radius < other_lo) or np.any(other_hi + radius < lo):
                continue
            distances, _ = other_tree.query(points)
            if np.mean(distances <= radius) >= .8:
                duplicate = True
                break
        if duplicate:
            keep[i] = False
            removed_nearby += 1
        else:
            accepted.append((regions[i], lo, hi, cKDTree(points)))
    return [c for c, yes in zip(chains, keep) if yes], {
        'removed_strays': removed_strays, 'removed_nearby': removed_nearby}
