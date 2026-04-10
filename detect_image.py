from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Input image path
image_path = "test.jpg"

# Read image
image = cv2.imread(image_path)

if image is None:
    print("❌ Error: Image not found")
    exit()

# Run detection
results = model(image)

# Annotated output
annotated_frame = results[0].plot()

# Show result
cv2.imshow("Image Detection", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()