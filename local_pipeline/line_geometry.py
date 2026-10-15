"""Geometry for conservative centerline cleanup and cubic SVG fitting."""

import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree


def resample(points, step=1.0):
    points = np.asarray(points, dtype=float)
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    keep = np.r_[True, np.diff(distance) > 1e-8]
    points, distance = points[keep], distance[keep]
    if len(points) < 2:
        return points
    samples = np.linspace(0, distance[-1], max(2, int(distance[-1] / step) + 1))
    return np.column_stack([np.interp(samples, distance, points[:, i]) for i in range(2)])


def remove_duplicates(chains, radius=1.25):
    """Only remove almost entirely overlapping paths; keep separated parallel lines."""
    if radius <= 0:
        return chains, 0
    ordered = sorted(chains, key=lambda c: -np.linalg.norm(np.diff(c, axis=0), axis=1).sum())
    kept, samples, trees = [], [], []
    lower = np.empty((len(ordered), 2))
    upper = np.empty((len(ordered), 2))
    removed = 0
    for chain in ordered:
        points = resample(chain)
        duplicate = False
        lo, hi = points.min(axis=0), points.max(axis=0)
        candidates = np.flatnonzero(np.all(hi + radius >= lower[:len(kept)], axis=1)
                                   & np.all(upper[:len(kept)] + radius >= lo, axis=1))
        for index in candidates:
            if trees[index] is None:
                trees[index] = cKDTree(samples[index])
            distances, _ = trees[index].query(points)
            if np.mean(distances <= radius) >= 0.95:
                duplicate = True
                break
        if duplicate:
            removed += 1
        else:
            lower[len(kept)], upper[len(kept)] = lo, hi
            kept.append(chain)
            samples.append(points)
            trees.append(None)
    return kept, removed


def bezier_bridge(p, q, outward_p, outward_q):
    distance = np.linalg.norm(q - p)
    control_a = p + outward_p * distance / 3
    control_b = q + outward_q * distance / 3
    t = np.linspace(0, 1, max(8, int(distance * 3) + 1))[:, None]
    return (1 - t)**3 * p + 3 * (1 - t)**2 * t * control_a + 3 * (1 - t) * t**2 * control_b + t**3 * q


class SegmentIndex:
    """Cache segment bounds while adding accepted bridges to the same geometry."""

    def __init__(self, chains):
        points = [np.asarray(c, dtype=float) for c in chains if len(c) >= 2]
        self.a = np.concatenate([c[:-1] for c in points]) if points else np.empty((0, 2))
        self.b = np.concatenate([c[1:] for c in points]) if points else np.empty((0, 2))
        self.lower = np.minimum(self.a, self.b)
        self.upper = np.maximum(self.a, self.b)

    def add(self, chain):
        points = np.asarray(chain, dtype=float)
        if len(points) < 2:
            return
        a, b = points[:-1], points[1:]
        self.a = np.concatenate((self.a, a))
        self.b = np.concatenate((self.b, b))
        self.lower = np.concatenate((self.lower, np.minimum(a, b)))
        self.upper = np.concatenate((self.upper, np.maximum(a, b)))

    def query(self, lower, upper):
        mask = np.all(self.upper >= lower, axis=1) & np.all(self.lower <= upper, axis=1)
        return self.a[mask], self.b[mask]


def bridge_crosses(bridge, chains):
    """Test sampled bridge segments against all existing segments, including new bridges."""
    def cross(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    lower, upper = bridge.min(axis=0) - 0.01, bridge.max(axis=0) + 0.01
    index = chains if isinstance(chains, SegmentIndex) else SegmentIndex(chains)
    a, b = index.query(lower, upper)
    if len(a):
        for p, q in zip(bridge[:-1], bridge[1:]):
            r, s = q - p, b - a
            denominator = cross(r, s)
            valid = np.abs(denominator) > 1e-9
            safe = np.where(valid, denominator, 1)
            t, u = cross(a - p, s) / safe, cross(a - p, r) / safe
            hit = valid & (t >= -1e-8) & (t <= 1 + 1e-8) & (u >= -1e-8) & (u <= 1 + 1e-8)
            intersections = p + t[hit, None] * r
            for intersection in intersections:
                if min(np.linalg.norm(intersection - bridge[0]), np.linalg.norm(intersection - bridge[-1])) > 0.75:
                    return True
            # Collinear overlaps are also collisions, excluding the endpoint neighborhoods.
            parallel = ~valid & (np.abs(cross(a - p, r)) < 1e-8)
            if np.any(parallel) and np.dot(r, r) > 1e-12:
                ta = ((a[parallel] - p) @ r) / np.dot(r, r)
                tb = ((b[parallel] - p) @ r) / np.dot(r, r)
                for lo, hi in zip(np.maximum(0, np.minimum(ta, tb)), np.minimum(1, np.maximum(ta, tb))):
                    if hi > lo + 1e-8:
                        midpoint = p + (lo + hi) / 2 * r
                        if min(np.linalg.norm(midpoint - bridge[0]), np.linalg.norm(midpoint - bridge[-1])) > 0.75:
                            return True
    return False


def spline_path(points, closed, tolerance=0.8, return_samples=False):
    """Fit the entire chain, constrain sampled error, and export exact cubic spans."""
    closed = bool(closed)
    points = resample(points)
    if len(points) < 4:
        path = 'M' + 'L'.join(f'{x:.3f} {y:.3f}' for x, y in points) + ('Z' if closed else '')
        return (path, points) if return_samples else path
    if closed and np.linalg.norm(points[0] - points[-1]) > 1e-8:
        points = np.vstack([points, points[0]])
    weights = np.ones(len(points))
    if not closed:
        weights[[0, -1]] = 1000
    budget = len(points) * tolerance**2
    point_tree = cKDTree(points)
    for _ in range(12):
        tck, u = splprep(points.T, w=weights, k=3, s=budget, per=closed)
        fitted = np.array(splev(u, tck)).T
        dense = np.array(splev(np.linspace(0, 1, max(40, len(points) * 4)), tck)).T
        # Dense samples limit overshoot between original samples as well.
        error = np.linalg.norm(fitted - points, axis=1).max()
        overshoot = point_tree.query(dense)[0].max()
        if error <= tolerance and overshoot <= tolerance + 0.55:
            break
        budget *= 0.35
    else:
        tck, _ = splprep(points.T, w=weights, k=3, s=0, per=closed)
    knots = np.unique(tck[0][3:-3])
    def xy(p):
        return f'{p[0]:.3f} {p[1]:.3f}'
    positions = np.array(splev(knots, tck)).T
    derivatives = np.array(splev(knots, tck, der=1)).T
    intervals = np.diff(knots)[:, None]
    controls_a = positions[:-1] + derivatives[:-1] * intervals / 3
    controls_b = positions[1:] - derivatives[1:] * intervals / 3
    spans = [f'C{xy(a)} {xy(b)} {xy(q)}' for a, b, q in zip(controls_a, controls_b, positions[1:])]
    path = 'M' + xy(positions[0]) + ''.join(spans) + ('Z' if closed else '')
    if return_samples:
        samples = np.array(splev(np.linspace(0, 1, max(40, len(points) * 3)), tck)).T
        return path, samples
    return path


def evidence_bridge(p, q, tangent_p, tangent_q, distance_map):
    """Choose an edge-supported cubic near the tangent bridge; reject unsupported gaps."""
    bridge = bezier_bridge(p, q, tangent_p, tangent_q)
    if distance_map is None:
        return bridge
    direction = q - p
    length = np.linalg.norm(direction)
    normal = np.array([-direction[1], direction[0]]) / max(length, 1e-8)
    t = np.linspace(0, 1, len(bridge))
    best = None
    for displacement in np.linspace(-min(3, length / 5), min(3, length / 5), 7):
        # This fourth-degree offset has zero endpoint position and derivative.
        trial = bridge + (16 * t**2 * (1 - t)**2 * displacement)[:, None] * normal
        ij = np.rint(trial).astype(int)
        y = np.clip(ij[:, 0], 0, distance_map.shape[0] - 1)
        x = np.clip(ij[:, 1], 0, distance_map.shape[1] - 1)
        distances = distance_map[y, x]
        if np.mean(distances <= 2.0) < 0.8 or distances.max() > 4:
            continue
        score = distances.mean() + abs(displacement) * 0.08
        if best is None or score < best[0]:
            best = score, trial
    return None if best is None else best[1]


def crossing_points(first, second=None):
    """Sampled polyline intersections; adjacent self segments are excluded."""
    first = np.asarray(first, dtype=float)
    self_check = second is None
    second = first if self_check else np.asarray(second, dtype=float)
    if len(first) < 2 or len(second) < 2:
        return np.empty((0, 2))
    if np.any(first.max(axis=0) < second.min(axis=0)) or np.any(second.max(axis=0) < first.min(axis=0)):
        return np.empty((0, 2))
    a, b = second[:-1], second[1:]
    delta = b - a
    lower, upper = np.minimum(a, b), np.maximum(a, b)
    first_lower = np.minimum(first[:-1], first[1:])
    first_upper = np.maximum(first[:-1], first[1:])
    intersections = []
    def cross(x, y):
        return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]
    # Bound temporary pair masks even for very long contours. Exact intersection
    # arithmetic is evaluated only for segments whose boxes overlap.
    batch = max(1, min(128, 262144 // len(a)))
    closed = self_check and np.linalg.norm(first[0] - first[-1]) < 1e-5
    for start in range(0, len(first) - 1, batch):
        stop = min(start + batch, len(first) - 1)
        lo, hi = first_lower[start:stop], first_upper[start:stop]
        mask = ((hi[:, None, 0] >= lower[None, :, 0])
                & (hi[:, None, 1] >= lower[None, :, 1])
                & (lo[:, None, 0] <= upper[None, :, 0])
                & (lo[:, None, 1] <= upper[None, :, 1]))
        if self_check:
            mask &= np.arange(len(a))[None, :] > np.arange(start, stop)[:, None] + 1
            if start == 0 and closed:
                mask[0, -1] = False
        rows, columns = np.nonzero(mask)
        if not len(rows):
            continue
        p = first[start + rows]
        r = first[start + rows + 1] - p
        aa, dd = a[columns], delta[columns]
        denominator = cross(r, dd)
        valid = np.abs(denominator) > 1e-9
        safe = np.where(valid, denominator, 1)
        t, u = cross(aa - p, dd) / safe, cross(aa - p, r) / safe
        hits = valid & (t >= -1e-8) & (t <= 1 + 1e-8) & (u >= -1e-8) & (u <= 1 + 1e-8)
        intersections.extend(p[hits] + t[hits, None] * r[hits])
    return np.unique(np.asarray(intersections).reshape(-1, 2).round(6), axis=0)


def safe_spline_paths(chains, tolerances):
    """Retry weaker smoothing when final curves introduce sampled crossings."""
    originals = [resample(c, 0.8) for c in chains]
    if not originals:
        return [], 0
    lower = np.array([c.min(axis=0) for c in originals])
    upper = np.array([c.max(axis=0) for c in originals])
    accepted, paths = [], []
    backed_off = 0
    for i, (chain, tolerance) in enumerate(zip(chains, tolerances)):
        baseline_self = crossing_points(originals[i])
        baseline_pairs = {}
        def new_crossing(hits, baseline):
            return len(hits) and (not len(baseline) or cKDTree(baseline).query(hits)[0].max() > 1)
        valid = False
        for factor in (1.0, 0.5, 0.2):
            closed = np.linalg.norm(np.asarray(chain[0]) - chain[-1]) < 1e-6
            path, samples = spline_path(chain, closed, tolerance * factor, True)
            if new_crossing(crossing_points(samples), baseline_self):
                continue
            valid = True
            candidates = np.flatnonzero(np.all(samples.max(axis=0) >= lower, axis=1)
                                       & np.all(upper >= samples.min(axis=0), axis=1))
            for j in candidates:
                if j == i:
                    continue
                obstacle = accepted[j] if j < i else originals[j]
                hits = crossing_points(samples, obstacle)
                if not len(hits):
                    continue
                if j not in baseline_pairs:
                    baseline_pairs[j] = crossing_points(originals[i], obstacle)
                if new_crossing(hits, baseline_pairs[j]):
                    valid = False
                    break
            if valid:
                if factor < 1:
                    backed_off += 1
                break
        if not valid:
            # Keep the measured polyline when every smooth fit introduces a collision.
            samples = np.asarray(chain)
            path = 'M' + 'L'.join(f'{x:.3f} {y:.3f}' for x, y in samples)
            if np.linalg.norm(samples[0] - samples[-1]) < 1e-6:
                path += 'Z'
            backed_off += 1
        accepted.append(samples)
        lower[i], upper[i] = samples.min(axis=0), samples.max(axis=0)
        paths.append(path)
    return paths, backed_off
