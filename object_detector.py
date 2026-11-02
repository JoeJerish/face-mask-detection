import cv2
import numpy as np
import torch
from ultralytics import YOLO


class ObjectDetector:
    """
    YOLOv8 Object Detector supporting multi-object detection & person tracking.
    Default imgsz=320 for ultra-fast 30+ FPS CPU execution.
    """

    def __init__(self, model_name="yolov8n.pt", conf_threshold=0.20, imgsz=640):
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[ObjectDetector] Loading {model_name} on device: {self.device} (imgsz={self.imgsz})...")
        self.model = YOLO(model_name)
        self.model.to(self.device)
        print(f"[ObjectDetector] YOLO model loaded successfully.")

    def set_confidence_threshold(self, conf):
        """Updates confidence threshold dynamically."""
        self.conf_threshold = max(0.10, min(0.95, conf))

    def set_image_size(self, imgsz):
        """Updates image inference size (320 for maximum speed, 640 for high resolution)."""
        if imgsz in [320, 480, 640]:
            self.imgsz = imgsz

    def switch_model(self, model_name):
        """Switches YOLO model dynamically."""
        if model_name != self.model_name:
            print(f"[ObjectDetector] Switching model to {model_name}...")
            self.model = YOLO(model_name)
            self.model.to(self.device)
            self.model_name = model_name

    def detect(self, frame):
        """
        Runs fast YOLO object detection on the input frame.
        """
        if frame is None or frame.size == 0:
            return {'detections': [], 'person_boxes': [], 'summary_counts': {}}

        # Run ultra-fast YOLO inference
        results = self.model(
            frame,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            iou=0.45,
            verbose=False,
            device=self.device
        )

        detections = []
        person_boxes = []
        summary_counts = {}

        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    cls_name = self.model.names.get(cls_id, f"class_{cls_id}").capitalize()

                    x1, y1, x2, y2 = map(int, xyxy)

                    # Noise filter
                    # Noise filter
                    # Ignore only extremely small detections
                    w, h = x2 - x1, y2 - y1
                    if w < 5 or h < 5:
                        continue

                    det_item = {
                        'box': (x1, y1, x2, y2),
                        'class_name': cls_name,
                        'confidence': round(conf * 100, 1),
                        'class_id': cls_id
                    }
                    detections.append(det_item)

                    summary_counts[cls_name] = summary_counts.get(cls_name, 0) + 1

                    if cls_name.lower() == 'person':
                        person_boxes.append((x1, y1, x2, y2))

        return {
            'detections': detections,
            'person_boxes': person_boxes,
            'summary_counts': summary_counts
        }
