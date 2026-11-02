import os
import base64
import cv2
import numpy as np

from flask import Flask, render_template, request, jsonify

from object_detector import ObjectDetector
from face_mask_detector import FaceMaskDetector


app = Flask(__name__, static_folder='static', template_folder='templates')

# Core AI Subsystems
print("[Backend] Initializing YOLO Object Detector...")
object_detector = ObjectDetector(conf_threshold=0.20, imgsz=640)

print("[Backend] Initializing Face & Mask Detector...")
mask_detector = FaceMaskDetector()

print("[Backend] AI Subsystems initialized successfully!")

# Global Camera & Telemetry State
# Global Telemetry State
latest_telemetry = {
    'detections': [],
    'person_boxes': [],
    'mask_results': [],
    'summary_counts': {},
    'fps': 0.0,
    'resolution': 'Browser Camera',
    'is_running': False,
    'speed_mode': 'balanced',
    'model_name': 'yolov8n.pt'
}





@app.route('/')
def index():
    return render_template('index.html')









@app.route('/api/live_data', methods=['GET'])
def get_live_data():
    """Returns current detection telemetry for UI dashboard & voice assistant."""
    return jsonify(latest_telemetry)


@app.route('/api/settings/speed', methods=['POST'])
def update_speed_mode():
    """Endpoint to update speed/accuracy mode ('fast', 'balanced', 'accurate')."""
    try:
        data = request.get_json()
        mode = data.get('mode', 'balanced')
        latest_telemetry['speed_mode'] = mode

        if mode == 'fast':
            object_detector.set_image_size(320)
        elif mode == 'balanced':
            object_detector.set_image_size(480)
        else:
            object_detector.set_image_size(640)

        return jsonify({'status': 'updated', 'mode': mode})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        'status': 'online',
        'yolo_model': object_detector.model_name,
        'face_mask_model': 'MobileNetV2 Fine-Tuned PyTorch',
        'device': str(object_detector.device),
        'speed_mode': latest_telemetry.get('speed_mode', 'balanced')
    })


@app.route('/api/detect', methods=['POST'])
def run_detection():

    try:

        data = request.get_json()

        if not data or 'image' not in data:
            return jsonify({
                'error': 'No image provided'
            }), 400

        img_data = data['image']

        if ',' in img_data:
            img_data = img_data.split(',', 1)[1]

        img_bytes = base64.b64decode(img_data)

        nparr = np.frombuffer(
            img_bytes,
            np.uint8
        )

        frame = cv2.imdecode(
            nparr,
            cv2.IMREAD_COLOR
        )

        if frame is None:
            return jsonify({
                'error': 'Invalid image format'
            }), 400

        conf_thresh = float(
            data.get('conf', 0.40)
        )
        print("[DEBUG] Frame received:", frame.shape)

        object_detector.set_confidence_threshold(
            conf_thresh
        )

        # YOLO detection
        det_results = object_detector.detect(
            frame
        )
        print("[DEBUG] YOLO result:", det_results)

        person_boxes = det_results.get(
            'person_boxes',
            []
        )
        print("[DEBUG] Person boxes:", person_boxes)

        # Face mask detection
        mask_results = (
            mask_detector.detect_and_classify_faces(
                frame,
                person_boxes
            )
        )
        print("[DEBUG] Mask results:", mask_results)

        # Update telemetry
        latest_telemetry.update({
            'detections':
                det_results.get(
                    'detections',
                    []
                ),

            'person_boxes':
                person_boxes,

            'summary_counts':
                det_results.get(
                    'summary_counts',
                    {}
                ),

            'mask_results':
                mask_results,

            'is_running':
                True,

            'resolution':
                f'{frame.shape[1]}x{frame.shape[0]}'
        })

        return jsonify({

            'detections':
                det_results.get(
                    'detections',
                    []
                ),

            'person_boxes':
                person_boxes,

            'summary_counts':
                det_results.get(
                    'summary_counts',
                    {}
                ),

            'mask_results':
                mask_results
        })

    except Exception as e:

        print(
            '[Detection Error]',
            str(e)
        )

        return jsonify({
            'error': str(e)
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860 if os.environ.get('SPACE_ID') else 5000))
    print("\n==================================================")
    print("   VISION ASSIST AI - WEB APPLICATION RUNNING    ")
    print(f"   Listening on port: {port}                    ")
    print("==================================================\n")

    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        threaded=True
    )

