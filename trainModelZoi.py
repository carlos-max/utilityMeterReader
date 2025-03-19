from ultralytics import YOLO

# Step 1: Initialize the Model
# Load a pre-trained YOLOv8 model (choose between yolov8n, yolov8s, yolov8m, etc.)
model = YOLO("yolov8n.pt")  # Nano model, small and fast for training

# Step 2: Train the Model
# Train the model on your custom dataset defined in 'dataMedicaoZoi.yaml'
model.train(
    data="dataMedicaoZoi.yaml",  # Path to your dataset configuration file
    epochs=50,         # Number of epochs to train
    imgsz=640,         # Input image size
    batch=16,          # Batch size
    name="medicao_zoi", # Experiment name
    workers=4          # Number of workers for data loading
)

# Step 3: Evaluate the Model
# Validate the model to check performance on the validation set
metrics = model.val()  # Automatically uses the validation data in 'dataMedicaoZoi.yaml'

# Print validation metrics
print("Validation Metrics:", metrics)

# Step 4: Run Inference
# Use the trained model to detect objects in new images or videos
results = model.predict(
    source="./input",  # Path to images/videos folder or single file
    save=True,                          # Save results to the 'runs' directory
    conf=0.25                           # Confidence threshold for predictions
)

# Step 5: Export the Model (Optional)
# Export the trained model to other formats like ONNX, CoreML, or TensorFlow
model.export(format="onnx")  # Replace 'onnx' with your desired format (e.g., 'tflite', 'torchscript')

print("Model export completed!")