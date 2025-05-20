import os
from PIL import Image
import torch
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import folder_paths
import numpy as np

class WatermarkDetectorLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}
    
    RETURN_TYPES = ("YOLO_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "watermark_detection"

    def load_model(self):
        model_path = os.path.join(folder_paths.models_dir, 'yolo')
        os.makedirs(model_path, exist_ok=True)
        file_path = os.path.join(model_path, "yolo11x-train28-best.pt")
        if not os.path.exists(file_path):
            hf_hub_download(repo_id="lrzjason/joy_caption_watermark_yolo", 
                          filename="yolo11x-train28-best.pt", 
                          local_dir=model_path)
        model = YOLO(file_path)
        return (model,)

class WatermarkDetector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("YOLO_MODEL",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "detect"
    CATEGORY = "watermark_detection"

    def detect(self, image, model, threshold=0.5):
        image = 255.0 * image[0].cpu().numpy()
        image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
        
        results = model(image, imgsz=1024, augment=True, iou=0.5, conf=threshold)
        assert len(results) == 1
        result = results[0]
        
        # Get bounding boxes
        bboxes = result.boxes.xyxy.cpu().numpy()
        
        # Create binary mask
        mask = Image.new('1', image.size, 0)
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            mask.paste(1, (x1, y1, x2, y2))
        
        # Convert to tensors
        mask_tensor = torch.from_numpy(np.array(mask)).float().unsqueeze(0)
        
        # Convert result image to tensor
        im_array = result.plot()
        result_image = Image.fromarray(im_array[..., ::-1])
        result_tensor = torch.from_numpy(np.array(result_image)).float() / 255.0
        result_tensor = result_tensor.unsqueeze(0)
        
        return (result_tensor, mask_tensor)

NODE_CLASS_MAPPINGS = {
    "WatermarkDetectorLoader": WatermarkDetectorLoader,
    "WatermarkDetector": WatermarkDetector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WatermarkDetectorLoader": "Load Watermark Detector",
    "WatermarkDetector": "Detect Watermarks",
}