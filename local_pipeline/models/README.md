# Face Landmarker

本目录的 `face_landmarker.task` 是下载的 Google MediaPipe Face Landmarker 官方模型，非本项目训练。

- 官方说明：https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
- 固定版本下载地址：https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
- 下载与校验摘要：运行 `python setup_models.py`。

模型仅在本机推理，不上传输入照片。请遵循官方模型的许可与使用条款。

SelfieMulticlass segmentation: https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite
SHA256: c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0
Size: 16371837 bytes. Official guide: https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter
