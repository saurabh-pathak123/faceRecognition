import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import time
model_path = 'blaze_face_short_range.tflite'

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
FaceDetectorResult = mp.tasks.vision.FaceDetectorResult
VisionRunningMode = mp.tasks.vision.RunningMode

results = []

def print_result(result: FaceDetectorResult, output_image: mp.Image, timestamp_ms: int):
    results.append(result)

options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

cam = cv2.VideoCapture(0)
with FaceDetector.create_from_options(options) as detector:
    while True:
        status, img = cam.read()
        ih, iw, _ = img.shape 
        if not status:
            print('Camera is not available')
            break
        img=cv2.flip(img,1)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        frame_timestamp_ms = int(time.time() * 1000)
        detector.detect_async(mp_image, frame_timestamp_ms)
        det = 0
        if results:
            result = results.pop()
            if result.detections:
                det = len(result.detections)
                
                for detection in result.detections:
                    x = detection.bounding_box.origin_x
                    y = detection.bounding_box.origin_y
                    w = detection.bounding_box.width
                    h = detection.bounding_box.height
                    cv2.rectangle(img, (x-20, y-20), (x+w+20, y+h+20), (0, 255, 0), 2)
                    for keypoints in detection.keypoints:  
                        nx = keypoints.x
                        ny = keypoints.y
                        x = int(nx * iw)
                        y = int(ny * ih) 
                        cv2.circle(img, (x, y), 2, (0, 0, 255), 2)
             
        if det:
            cv2.rectangle(img, (0,0), (iw, 50), (100,255,100), -1)
            cv2.putText(img, f'{det} faces detected', (30,30), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
        else:
            cv2.rectangle(img, (0,0), (iw, 50), (100,100,255), -1)
            cv2.putText(img, f'No faces detected', (30,30), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
 
        cv2.imshow('Face Detection', img)
        if cv2.waitKey(1) == 27: # ESC KEY
            break
    cam.release()
    cv2.destroyAllWindows()