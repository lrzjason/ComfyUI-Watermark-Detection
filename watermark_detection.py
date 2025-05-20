import gradio as gr
from PIL import Image
from ultralytics import YOLO
import torchvision.transforms.functional as TVF
from transformers import Owlv2VisionModel
from torch import nn
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
import os

def yolo_predict(image: Image.Image) -> tuple:
	results = yolo_model(image, imgsz=1024, augment=True, iou=0.5)
	assert len(results) == 1
	result = results[0]
	im_array = result.plot()
	im = Image.fromarray(im_array[..., ::-1])
	
	# Get bounding boxes
	bboxes = result.boxes.xyxy.cpu().numpy()
	
	# Create binary mask
	mask = Image.new('1', image.size, 0)
	for bbox in bboxes:
		x1, y1, x2, y2 = map(int, bbox)
		mask.paste(1, (x1, y1, x2, y2))
	
	return im, mask


def predict(image: Image.Image, conf_threshold: float):
	# YOLO
	yolo_image, mask = yolo_predict(image)

	return yolo_image, mask

file_path = "yolo11x-train28-best.pt"
if not os.path.exists(file_path):
	hf_hub_download(repo_id="lrzjason/joy_caption_watermark_yolo", filename="yolo11x-train28-best.pt", local_dir="./")

# Load YOLO model
yolo_model = YOLO(file_path)

gradio_app = gr.Blocks()
with gr.Blocks() as app:
	gr.HTML(
		"""
		<h1>Watermark Detection</h1>
		"""
	)

	with gr.Row():
		with gr.Column():
			image = gr.Image(type="pil", label="Image")
			conf_threshold = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, label="Confidence Threshold")
			btn_submit = gr.Button(value="Detect Watermarks")
		
		with gr.Column():
			image_yolo = gr.Image(type="pil", label="YOLO Detections")
			image_mask = gr.Image(type="pil", label="Binary Mask")
		

	btn_submit.click(fn=predict, inputs=[image, conf_threshold], outputs=[image_yolo, image_mask])


if __name__ == "__main__":
	app.launch()