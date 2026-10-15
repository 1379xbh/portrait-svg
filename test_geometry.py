"""Synthetic regressions for identity protection, budgets and vector metrics."""
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / 'local_pipeline'))
from line_cleanup import budget_chains
from portrait_regions import glasses_evidence, protect_glasses
from informative import vectorize_drawing
from svg_io import svg_and_json, svg_metrics


class GeometryTests(unittest.TestCase):
    def fixture(self):
        gray = np.full((100, 100), 255, np.uint8)
        labels = np.ones_like(gray)
        chains = [[(y, x) for x in range(a, b)] for y, a, b in
                  [(20, 10, 30), (35, 10, 70), (50, 10, 40), (65, 10, 50)]]
        for chain in chains:
            for y, x in chain:
                gray[y, x] = 0
            # Additional ink on either side makes these internal strokes.
            gray[chain[0][0], [0, 99]] = 0
        gray[[0, 99], :] = 0
        labels[20, 10:30] = 10
        return gray, labels, chains

    def test_budget_keeps_face_and_best_supported_long_stroke(self):
        gray, labels, chains = self.fixture()
        selected, report = budget_chains(chains, labels, gray, 1)
        self.assertEqual(selected, chains[:2])
        self.assertEqual(report['removed_by_budget'], 2)
        selected, _ = budget_chains(chains, labels, gray, 0)
        self.assertEqual(selected, chains[:1])

    def test_no_face_does_not_arbitrarily_delete_identity(self):
        gray, _, chains = self.fixture()
        selected, _ = budget_chains(chains, None, gray, 0)
        self.assertEqual(selected, chains)

    def test_boundary_and_connectors_survive_zero_budget(self):
        gray, labels, _ = self.fixture()
        chains = [[(80, x) for x in range(10, 70)],
                  [(30, 30), (31, 30), (32, 30)],
                  [(28, 30), (29, 30), (30, 30)],
                  [(32, 30), (33, 30), (34, 30)]]
        gray[81:, :] = 255
        for y, x in chains[0]:
            gray[y, x] = 0
        selected, _ = budget_chains(chains, labels, gray, 0)
        self.assertIn(chains[0], selected)
        self.assertIn(chains[1], selected)

    def test_metrics_exclude_definitions_and_background(self):
        svg = '<svg viewBox="0 0 100 100"><defs><clipPath id="e"><path d="M0 0L2 0L2 2Z"/></clipPath></defs><path fill="white" d="M0 0H100V100H0Z"/><g fill="none" stroke="#111"><path d="M0 0L10 0M20 0C21 1 22 1 23 0"/><circle cx="5" cy="5" r="1"/></g></svg>'
        report = svg_metrics(svg)
        self.assertEqual(report['visible_paths'], 1)
        self.assertEqual(report['visible_strokes'], 3)
        self.assertEqual(report['segments'], 2)
        self.assertEqual(report['curve_segments'], 1)
        self.assertGreater(report['path_length'], 13)

    def test_closed_glasses_ring_triggers_evidence(self):
        gray = np.full((100, 120), 255, np.uint8)
        points = np.zeros((478, 2))
        points[33] = points[133] = (40, 40)
        points[263] = points[362] = (40, 85)
        regions = {'points': points, 'face_width': 100}
        self.assertFalse(glasses_evidence(gray, regions))
        cv2.rectangle(gray, (23, 29), (57, 51), 0, 2)
        self.assertTrue(glasses_evidence(gray, regions))
        labels = np.full_like(gray, 4)
        labels[40, 40] = 10
        protected = protect_glasses(labels, regions)
        self.assertEqual(protected[40, 40], 10)
        self.assertEqual(protected[40, 60], 16)
        self.assertEqual(protected[90, 60], 4)

    def test_contour_without_face_falls_back_and_generates_valid_svg(self):
        with tempfile.TemporaryDirectory() as folder:
            drawing = Path(folder) / 'drawing.png'
            photo = Path(folder) / 'photo.png'
            output = Path(folder) / 'result.svg'
            gray = np.full((80, 80), 255, np.uint8)
            cv2.line(gray, (10, 10), (60, 60), 0, 2)
            Image.fromarray(gray).save(drawing)
            Image.fromarray(gray).convert('RGB').save(photo)
            with patch('informative.detect_regions', return_value=None):
                report = vectorize_drawing(drawing, output, portrait_source=photo,
                                           detail='contour', detail_budget=0)
            self.assertEqual(report['fallback_reason'], 'face_not_detected')
            svg_and_json(output.read_text())


if __name__ == '__main__':
    unittest.main()
