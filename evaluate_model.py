from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, average_precision_score

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# ===== DEFINE YOUR TEST DATA =====
# Replace with your image paths + ground truth
data = [
    {"path": "test.jpg", "gt": [0]},       # person
    {"path": "test2.jpg", "gt": [2]},      # car
    {"path": "test3.jpg", "gt": [0, 2]}    # person, car
]

y_true = []
y_pred = []
y_scores = []

# ===== PROCESS IMAGES =====
for item in data:
    img_path = item["path"]
    gt_classes = item["gt"]

    image = cv2.imread(img_path)

    if image is None:
        print(f"❌ Image not found: {img_path}")
        continue

    results = model(image)[0]

    if results.boxes is not None and len(results.boxes) > 0:
        pred_classes = results.boxes.cls.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()

        print(f"\n📸 {img_path}")
        print("GT:", gt_classes)
        print("Pred:", pred_classes)

        # Match predictions with ground truth (simple pairing)
        for i in range(min(len(gt_classes), len(pred_classes))):
            y_true.append(gt_classes[i])
            y_pred.append(int(pred_classes[i]))
            y_scores.append(scores[i])
    else:
        print(f"⚠️ No detection in {img_path}")

# ===== CHECK DATA =====
if len(y_true) == 0:
    print("❌ No valid data for evaluation")
    exit()

print("\nFinal y_true:", y_true)
print("Final y_pred:", y_pred)

# ===== CONFUSION MATRIX =====
cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

# ===== CONVERT TO BINARY (Correct vs Wrong) =====
y_true_binary = []
y_scores_binary = []

for t, p, s in zip(y_true, y_pred, y_scores):
    if t == p:
        y_true_binary.append(1)   # correct detection
    else:
        y_true_binary.append(0)   # wrong detection
    
    y_scores_binary.append(s)

# ===== PRECISION-RECALL CURVE =====
precision, recall, _ = precision_recall_curve(y_true_binary, y_scores_binary)

plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()

# ===== mAP =====
map_score = average_precision_score(y_true_binary, y_scores_binary)
print(f"\n✅ mAP: {map_score:.4f}")
