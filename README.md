# Car Counter using YOLO and OpenCV

This project demonstrates real-time car counting using the YOLO (You Only Look Once) algorithm and OpenCV (cv2).

## Features

- Real-time car counting using a webcam or video feed
- Utilizes YOLO for high-speed detection
- Uses a region close to a specified line to detect cars and increment the counter
- Simple to set up and run

## Requirements

- Python 3.x
- OpenCV
- NumPy
- YOLO weights and configuration files

## Installation

1. Clone this repository:
    ```sh
    git clone https://github.com/Aymanquad/Car-Counter-Object-Detection-.git
    cd car-counter
    ```

2. Install the required packages:
    ```sh
    pip install opencv-python numpy
    ```

3. Download YOLO weights and configuration files:
    - YOLOv3 weights: [YOLOv3.weights](https://pjreddie.com/media/files/yolov3.weights)
    - YOLOv3 config: [yolov3.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)
    - COCO names: [coco.names](https://github.com/pjreddie/darknet/blob/master/data/coco.names)

4. Place the downloaded files in the project directory.

## Usage

Run the car counting script:
```sh
python car-counter.py
