## Object Detection System using YOLOv8

This project is a real-time object detection system built using YOLOv8. It supports detection on images, videos, and live webcam feeds.

## Features
🔍 Real-time object detection (Webcam)
🎥 Video file detection support
🖼️ Image detection
📦 Pre-trained YOLOv8 model (COCO dataset)
🏷️ Bounding boxes with class labels
⚡ Fast and efficient detection

## Tech Stack
Python
OpenCV
YOLOv8 (Ultralytics)
NumPy
PyTorch


📁Project Structure
object-detection-project/
│── detect_webcam.py
│── detect_video.py
│── detect_image.py
│── requirements.txt
│── README.md
│── outputs/


⚙️ Installation
Clone the repository:
git clone https://github.com/your-username/object-detection-project.git
cd object-detection-project

Install dependencies:
pip install -r requirements.txt

▶️ Usage
🔴 Run Webcam Detection
python detect_webcam.py
🎥 Run Video Detection
python detect_video.py
🖼️ Run Image Detection
python detect_image.py

📊 Dataset
This project uses the COCO dataset, which includes 80 object classes like:
Person
Car
Dog
Chair
Bottle

📸 Output
Displays bounding boxes around detected objects
Shows class labels and confidence scores
Can be extended to save output images/videos

## Future Improvements
✅ Object counting system
🚨 Alert system for specific objects
🎯 Custom dataset training
📊 Performance metrics (FPS, accuracy)

## Author
Anmol Kathayat
Prabin Dahal

## Contribute
Feel free to fork this project and improve it!

## License
This project is open-source and free to use.