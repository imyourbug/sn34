import time
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

# Load model and processor
processor = AutoImageProcessor.from_pretrained("dima806/deepfake_vs_real_image_detection")
model = AutoModelForImageClassification.from_pretrained("dima806/deepfake_vs_real_image_detection")

# Load an image
image = Image.open("images/test.png")
inputs = processor(images=image, return_tensors="pt")

# Measure inference time
start_time = time.time()
outputs = model(**inputs)  # Run on CPU
print(f"outputs: {outputs}")
