# ComfyUI Watermark Detection Node

This custom node for ComfyUI provides watermark detection capabilities using a YOLO model trained by [fancyfeast](https://huggingface.co/fancyfeast), the creator of JoyCaption. The model is originally hosted at [Hugging Face Space](https://huggingface.co/spaces/fancyfeast/joycaption-watermark-detection).

## Features
- Detects watermarks in images using YOLOv11
- Returns both the detection visualization and a binary mask
- Adjustable confidence threshold for detections

## Installation
1. Place this folder in your `ComfyUI/custom_nodes/` directory
2. The model will be automatically downloaded from Hugging Face on first use

## Usage
1. Add a `Load Watermark Detector` node to load the model
2. Connect your image to the `Detect Watermarks` node
3. Adjust the confidence threshold as needed (default: 0.5)
4. The node outputs:
   - Detection visualization (IMAGE)
   - Binary mask of detected watermarks (MASK)

## Model Details
- Model: YOLOv11 trained on watermark detection
- Original Author: fancyfeast (JoyCaption creator)
- Model Source: [Hugging Face](https://huggingface.co/spaces/fancyfeast/joycaption-watermark-detection)
- Re-hosted at [Hugging Face](https://huggingface.co/lrzjason/joy_caption_watermark_yolo)

## Notes
- The model will be downloaded to `ComfyUI/models/yolo/` automatically
- Higher confidence thresholds will result in fewer but more certain detections