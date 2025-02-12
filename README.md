# Real-time Object Detection

A computer vision project that performs real-time object detection using YOLO (You Only Look Once) and OpenCV.

## Description

This project uses YOLOv3 to detect and classify objects in real-time through your webcam. It can detect 80 different classes of objects including people, vehicles, animals, and everyday items.

## Prerequisites

- Python 3.x
- Required packages (install via requirements.txt):
  - opencv-contrib-python==4.11.0.86
  - cvlib==0.2.7
  - gtts==2.5.4
  - playsound==1.3.0

## Required Files

- `yolov3.cfg` - YOLOv3 configuration file
- `yolov3.weights` - YOLOv3 pre-trained weights
- `coco.names` - List of object classes that can be detected

## Installation

1. Clone this repository:
```sh
git clone https://github.com/yourusername/object_detection.git
cd object_detection
```
2. Install the required packages:
```sh
pip install -r requirements.txt
```
3. Download YOLOv3 files:
   - Get `yolov3.weights` from: https://pjreddie.com/media/files/yolov3.weights
   - Get `yolov3.cfg` from: https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
   - Get `coco.names` from: https://github.com/pjreddie/darknet/blob/master/data/coco.names

## Usage

1. Place all downloaded YOLOv3 files in the project directory
2. Run the main script:
```sh
python main.py
```
### Controls
- Press 'q' to quit the application

## Features

- Real-time object detection using webcam feed
- Support for 80 different object classes
- Adjustable confidence threshold (default: 0.3)
- Color-coded bounding boxes for different object classes
- Real-time display of confidence scores

## Technical Details

- Input resolution: 416x416 pixels
- Model: YOLOv3 (You Only Look Once version 3)
- Backend: OpenCV DNN module
- Video capture: Real-time webcam feed

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## Contact Information

- For more information please contact the author at ammaryasser12603@gmail.com.
