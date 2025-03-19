from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO("model/weights.h5")

# Perform inference on a single image
results = model.predict(
    source="./1.cropped_box_0.jpg",  # Path to the input image
    conf=0.25,                  # Confidence threshold (default is 0.25)
    save=True    # Save the results in the `runs/predict` directory
)




