import cv2
import mediapipe as mp
import time

# Simplified imports
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Initialize webcam
cap = cv2.VideoCapture(0)

# 1. Change running_mode to VIDEO
options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='model/blaze_face_short_range.tflite'),
    running_mode=VisionRunningMode.VIDEO 
)

with FaceDetector.create_from_options(options) as detector:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # MediaPipe requires RGB images
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # 2. Use detect_for_video and a real timestamp
        # Using time.time() ensures a monotonically increasing timestamp
        timestamp_ms = int(time.time() * 1000)
        result = detector.detect_for_video(mp_image, timestamp_ms)

        # 3. Process the results
        if result.detections:
            for detection in result.detections:
                bbox = detection.bounding_box
                x, y, w, h = int(bbox.origin_x), int(bbox.origin_y-40), int(bbox.width), int(bbox.height+50)
                
                # Boundary checks to prevent crashing if face is partially off-screen
                H, W, _ = frame.shape
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(W, x + w), min(H, y + h)
                
                if x2 > x1 and y2 > y1:
                    face_roi = frame[y1:y2, x1:x2]
                    # Apply strong blur
                    blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
                    frame[y1:y2, x1:x2] = blurred_face

        # 4. Show the frame (Moved outside the 'if detections' block)
        cv2.imshow('Face Anonimazer', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()