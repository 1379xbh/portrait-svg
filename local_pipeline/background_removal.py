"""CPU portrait segmentation. Retains the original canvas and facial pixels."""
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image, ImageOps


def clean_mask(confidence):
    binary = (confidence >= .5).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        raise ValueError('No person found by segmentation')
    foreground = (labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    return foreground * 255


def remove_background(source, destination, size=800):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    started = time.perf_counter()
    model = Path(__file__).parent / 'models/selfie_multiclass_256x256.tflite'
    with Image.open(source) as image:
        photo = ImageOps.exif_transpose(image).convert('RGB')
        photo.thumbnail((size, size), Image.Resampling.LANCZOS)
    rgb = np.asarray(photo)
    options = vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_buffer=model.read_bytes(), delegate=BaseOptions.Delegate.CPU),
        output_confidence_masks=True)
    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        result = segmenter.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        confidence = 1 - result.confidence_masks[0].numpy_view().copy()
    if confidence.shape != rgb.shape[:2]:
        confidence = cv2.resize(confidence, photo.size)
    mask = clean_mask(confidence)
    # Feather only the silhouette; keep interior pixels, including glasses, intact.
    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), .65)
    composite = np.rint(rgb * alpha[..., None] + 255 * (1-alpha[..., None])).astype(np.uint8)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(composite).save(destination)
    mask_path = destination.with_name(destination.stem + '_mask.png')
    Image.fromarray(mask).save(mask_path)
    return mask, {'background_model': model.name, 'background_device': 'cpu',
                  'background_seconds': round(time.perf_counter()-started, 3),
                  'white_background_photo': str(destination.resolve()),
                  'person_mask': str(mask_path.resolve())}
