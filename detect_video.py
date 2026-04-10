from ultralytics import YOLO
import cv2
import time

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# ===== SELECT INPUT SOURCE =====
print("Select input source:")
print("1 - Webcam")
print("2 - Video file")

choice = input("Enter choice (1/2): ")

if choice == "1":
    cap = cv2.VideoCapture(0)
elif choice == "2":
    video_path = input("Enter video file path: ")
    cap = cv2.VideoCapture(video_path)
else:
    print("❌ Invalid choice")
    exit()

if not cap.isOpened():
    print("❌ Error opening video source")
    exit()

# ===== SETTINGS =====
line_y = 300
counted_ids = set()
total_count = 0

# 👉 Set to None to count all objects
# Example: 2 = car, 0 = person
TARGET_CLASS = None  

# FPS calc
prev_time = 0

# ===== PROCESS VIDEO =====
while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ Video ended")
        break

    # Run tracking
    results = model.track(frame, persist=True)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy
        ids = results[0].boxes.id
        classes = results[0].boxes.cls

        for box, obj_id, cls in zip(boxes, ids, classes):
            x1, y1, x2, y2 = map(int, box)
            obj_id = int(obj_id)
            cls = int(cls)

            # Filter class if needed
            if TARGET_CLASS is not None and cls != TARGET_CLASS:
                continue

            # Center point
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)

            # Count logic
            if cy > line_y and obj_id not in counted_ids:
                counted_ids.add(obj_id)
                total_count += 1

    # Draw counting line
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (255,0,0), 2)

    # Show count
    cv2.putText(frame, f"Count: {total_count}", (20,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    # ===== FPS =====
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}", (20,90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    # Show frame
    cv2.imshow("Object Detection & Counting", frame)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

# pip install ultralytics opencv-python
# python.exe -m pip install --upgrade pip