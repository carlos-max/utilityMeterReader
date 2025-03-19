import os
import cv2
from ultralytics import YOLO

# Load the trained model
model = YOLO("model/medicaoZoi.pt")

# Folder containing the images to process
input_folder = "./input/inputZoi"
output_folder = "./input/inputNumber"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Iterate through all the image files in the input folder
for image_name in os.listdir(input_folder):
    # Skip files that are not images (e.g., directories or other file types)
    if not image_name.endswith(('.jpg', '.png', '.jpeg')):
        continue

    # Construct the full image path
    image_path = os.path.join(input_folder, image_name)

    # Perform inference on the image
    results = model.predict(
        source=image_path,  # Path to the input image
        conf=0.25,  # Confidence threshold (default is 0.25)
    )

    # Load the image for cropping
    image = cv2.imread(image_path)

    # Extract bounding box predictions and crop the image
    for i, result in enumerate(results[0].boxes.xywhn):  # Normalized format (x_center, y_center, w, h)
        # Get bounding box coordinates in pixel format
        x_center, y_center, width, height = result.cpu().numpy()
        x_center *= image.shape[1]
        y_center *= image.shape[0]
        width *= image.shape[1]
        height *= image.shape[0]

        # Convert to corner coordinates
        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        x_max = int(x_center + width / 2)
        y_max = int(y_center + height / 2)

        # Crop the image
        cropped_image = image[y_min:y_max, x_min:x_max]

        # Save the cropped image with a unique name
        save_path = os.path.join(output_folder, f"cropped_{image_name}_{i}.jpg")
        cv2.imwrite(save_path, cropped_image)
        print(f"Saved: {save_path}")