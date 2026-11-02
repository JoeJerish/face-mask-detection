import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import os
from collections import deque


class MaskNet(nn.Module):
    """MobileNetV2 based Face Mask Classifier matching pretrained weights."""
    def __init__(self):
        super(MaskNet, self).__init__()
        base_model = models.mobilenet_v2(weights=None)
        self.features = base_model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(base_model.last_channel, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # Index 0: MASK ON, Index 1: MASK OFF
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class FaceMaskDetector:
    """
    High-accuracy, ultra-fast Face & Mask Detector.
    Uses PyTorch MobileNetV2 with YCrCb color space + lower-face feature analysis.
    """

    def __init__(self, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[FaceMaskDetector] Initializing engine on device: {self.device}")

        # Load OpenCV Frontal Face Cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # Image transform for PyTorch classifier
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Initialize PyTorch Mask Model
        self.model = MaskNet().to(self.device)
        self._load_weights()
        self.model.eval()

        # Prediction smoothing buffer per grid cell
        self.history = {}

    def _load_weights(self):
        """Loads fine-tuned weights if available."""
        weight_path = os.path.join(os.path.dirname(__file__), "models", "face_mask_mobilenetv2.pth")
        if os.path.exists(weight_path):
            try:
                state_dict = torch.load(weight_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"[FaceMaskDetector] Loaded trained weights from {weight_path}")
                return
            except Exception as e:
                print(f"[FaceMaskDetector] Could not load weight dict: {e}")

        # Default initialization
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def detect_and_classify_faces(self, frame, person_boxes=None):
        """
        Fast multi-scale face detection & accurate mask classification.
        """
        if frame is None or frame.size == 0:
            return []

        results = []
        h_img, w_img = frame.shape[:2]

        # Fast 0.25x downscaled gray frame for sub-5ms cascade detection
        scale = 0.25
        small_gray = cv2.cvtColor(cv2.resize(frame, (int(w_img * scale), int(h_img * scale))), cv2.COLOR_BGR2GRAY)

        detected_faces = []

        # 1. Search inside person crops if available
        if person_boxes:
            for p_box in person_boxes:
                px1, py1, px2, py2 = map(int, p_box)
                spx1, spy1 = int(max(0, px1) * scale), int(max(0, py1) * scale)
                spx2, spy2 = int(min(w_img, px2) * scale), int(min(h_img, py2) * scale)
                srw, srh = spx2 - spx1, spy2 - spy1

                if srw < 15 or srh < 15:
                    continue

                roi_small = small_gray[spy1:spy2, spx1:spx2]
                faces = self.face_cascade.detectMultiScale(
                    roi_small, scaleFactor=1.2, minNeighbors=3, minSize=(12, 12)
                )

                for (fx, fy, fw, fh) in faces:
                    abs_x = int((spx1 + fx) / scale)
                    abs_y = int((spy1 + fy) / scale)
                    abs_w = int(fw / scale)
                    abs_h = int(fh / scale)
                    detected_faces.append((abs_x, abs_y, abs_w, abs_h))

        # 2. Fallback full-frame search if no person boxes provided
        if not detected_faces:
            faces = self.face_cascade.detectMultiScale(
                small_gray, scaleFactor=1.25, minNeighbors=4, minSize=(15, 15)
            )
            for (fx, fy, fw, fh) in faces:
                abs_x = int(fx / scale)
                abs_y = int(fy / scale)
                abs_w = int(fw / scale)
                abs_h = int(fh / scale)
                detected_faces.append((abs_x, abs_y, abs_w, abs_h))

        # 3. Fallback: Upper portion of person box if cascade misses face
        if not detected_faces and person_boxes:
            for p_box in person_boxes:
                px1, py1, px2, py2 = map(int, p_box)
                pw, ph = px2 - px1, py2 - py1
                if pw > 40 and ph > 60:
                    face_h = int(ph * 0.35)
                    face_w = int(pw * 0.70)
                    fx = px1 + int((pw - face_w) / 2)
                    fy = py1 + int(ph * 0.04)
                    detected_faces.append((max(0, fx), max(0, fy), face_w, face_h))

        # Classify each face ROI
        if detected_faces:
            face_crops = []
            valid_boxes = []

            for (fx, fy, fw, fh) in detected_faces:
                fx1, fy1 = max(0, fx), max(0, fy)
                fx2, fy2 = min(w_img, fx + fw), min(h_img, fy + fh)
                if (fx2 - fx1) < 20 or (fy2 - fy1) < 20:
                    continue
                crop = frame[fy1:fy2, fx1:fx2]
                if crop.size > 0:
                    face_crops.append(crop)
                    valid_boxes.append((fx1, fy1, fx2, fy2))

            if face_crops:
                predictions = self._predict_mask_batch(face_crops)
                for (fx1, fy1, fx2, fy2), (status, conf) in zip(valid_boxes, predictions):
                    smoothed_status, smoothed_conf = self._apply_temporal_smoothing(fx1, fy1, status, conf)
                    results.append({
                        'box': (fx1, fy1, fx2, fy2),
                        'status': smoothed_status,
                        'confidence': smoothed_conf
                    })

        return results

    def _predict_mask_batch(self, face_crops):
        """Batch predicts face mask status combining PyTorch MobileNetV2 + YCrCb Skin & Edge Analysis."""
        results = []
        if not face_crops:
            return results

        tensors = []
        feature_scores = []

        for crop in face_crops:
            rgb_face = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(rgb_face))
            feature_scores.append(self._analyze_lower_face_features(crop))

        batch_tensor = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            outputs = self.model(batch_tensor)
            probs = torch.softmax(outputs, dim=1)

        for i in range(len(face_crops)):
            # Model prediction: index 0 = MASK ON, index 1 = MASK OFF
            nn_mask_on_prob = probs[i][0].item()
            nn_mask_off_prob = probs[i][1].item()

            feat_status, feat_conf = feature_scores[i]

            if feat_status == "MASK OFF" and feat_conf >= 70.0:
                # Exposed mouth/lips skin tone -> MASK OFF
                final_status = "MASK OFF"
                final_conf = max(feat_conf, round(nn_mask_off_prob * 100, 1))
            elif feat_status == "MASK ON" and feat_conf >= 70.0:
                # Mask covering lower face -> MASK ON
                final_status = "MASK ON"
                final_conf = max(feat_conf, round(nn_mask_on_prob * 100, 1))
            else:
                # Ensemble decision
                if nn_mask_off_prob > nn_mask_on_prob:
                    final_status = "MASK OFF"
                    final_conf = round(nn_mask_off_prob * 100, 1)
                else:
                    final_status = "MASK ON"
                    final_conf = round(nn_mask_on_prob * 100, 1)

            results.append((final_status, min(99.0, max(50.0, final_conf))))

        return results

    def _analyze_lower_face_features(self, face_crop):
        """
        Analyzes lower face area for exposed skin tone (YCrCb) and edge/mouth details.
        Returns ("MASK ON" | "MASK OFF" | "UNKNOWN", confidence)
        """
        h, w = face_crop.shape[:2]
        if h < 15 or w < 15:
            return "MASK OFF", 50.0

        lower_face = face_crop[int(h * 0.45):int(h * 0.95), int(w * 0.15):int(w * 0.85)]
        if lower_face.size == 0:
            return "MASK OFF", 50.0

        # 1. YCrCb Skin Color Space Analysis
        ycrcb = cv2.cvtColor(lower_face, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        skin_pixels = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
        skin_ratio = np.count_nonzero(skin_pixels) / float(skin_pixels.size + 1e-6)

        # 2. Edge / Contour Detail
        gray_lower = cv2.cvtColor(lower_face, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray_lower, cv2.CV_64F).var()

        # Decision matrix:
        # High skin ratio in YCrCb (>28%) -> MASK OFF (unmasked human face)
        if skin_ratio >= 0.28:
            return "MASK OFF", round(min(98.0, 75.0 + skin_ratio * 30.0), 1)
        elif skin_ratio <= 0.15 and laplacian_var < 220.0:
            # Low skin coverage + covered lower face -> MASK ON
            return "MASK ON", round(min(98.0, 85.0 - skin_ratio * 50.0), 1)
        else:
            return "UNKNOWN", 50.0

    def _apply_temporal_smoothing(self, x, y, status, conf):
        """Smooths predictions across 3 frames to avoid erratic toggling."""
        grid_key = (x // 80, y // 80)
        prob_val = conf / 100.0 if status == "MASK ON" else (1.0 - conf / 100.0)

        if grid_key not in self.history:
            self.history[grid_key] = deque(maxlen=3)

        self.history[grid_key].append(prob_val)
        avg_prob = sum(self.history[grid_key]) / len(self.history[grid_key])

        if avg_prob >= 0.50:
            return "MASK ON", round(avg_prob * 100, 1)
        else:
            return "MASK OFF", round((1.0 - avg_prob) * 100, 1)
