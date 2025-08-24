import cv2
import numpy as np
from ultralytics import YOLO
import time
import argparse
import os

class YOLODetector:
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.5, iou_threshold=0.45):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        # Check if model exists, download if not
        if not os.path.exists(model_path):
            print(f"Model {model_path} not found. Downloading...")
            try:
                model = YOLO(model_path)  # This will download the model
                print("Download completed!")
            except Exception as e:
                print(f"Error downloading model: {e}")
                return
        
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # COCO class names (80 classes)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
        # Colors for different classes
        np.random.seed(42)
        self.colors = np.random.uniform(0, 255, size=(len(self.class_names), 3))
    
    def detect_objects(self, frame):
        """
        Detect objects in a frame
        
        Args:
            frame: Input image/frame
            
        Returns:
            detections: List of detections with format [x1, y1, x2, y2, conf, class_id]
        """
        results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    detections.append([x1, y1, x2, y2, conf, class_id])
        
        return detections
    
    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame: Input frame
            detections: List of detections
            
        Returns:
            annotated_frame: Frame with drawn detections
        """
        annotated_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2, conf, class_id = detection
            
            # Get class name and color
            class_name = self.class_names[class_id]
            color = self.colors[class_id]
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, 
                         (int(x1), int(y1)), 
                         (int(x2), int(y2)), 
                         color, 2)
            
            # Prepare label
            label = f"{class_name}: {conf:.2f}"
            
            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Draw label background
            cv2.rectangle(annotated_frame,
                         (int(x1), int(y1) - text_height - 10),
                         (int(x1) + text_width, int(y1)),
                         color, -1)
            
            # Draw label text
            cv2.putText(annotated_frame, label,
                       (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (255, 255, 255), 2)
        
        return annotated_frame

class RealTimeYOLO:
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.5):
        self.detector = YOLODetector(model_path, conf_threshold)
        self.fps_counter = 0
        self.start_time = time.time()
    
    def detect_from_webcam(self, camera_id=0):
        """
        Real-time detection from webcam
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Press 'q' to quit, 's' to save current frame")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects
            detections = self.detector.detect_objects(frame)
            
            # Draw detections
            annotated_frame = self.detector.draw_detections(frame, detections)
            
            # Calculate and display FPS
            self.fps_counter += 1
            elapsed_time = time.time() - self.start_time
            
            if elapsed_time >= 1.0:
                fps = self.fps_counter / elapsed_time
                self.fps_counter = 0
                self.start_time = time.time()
            else:
                fps = self.fps_counter / elapsed_time if elapsed_time > 0 else 0
            
            # Add FPS to frame
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add detection count
            cv2.putText(annotated_frame, f"Objects: {len(detections)}",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('YOLO Real-time Detection', annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = int(time.time())
                filename = f'detection_save_{timestamp}.jpg'
                cv2.imwrite(filename, annotated_frame)
                print(f"Frame saved as {filename}!")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def detect_from_video(self, video_path, output_path=None):
        """
        Process video file
        """
        if not os.path.exists(video_path):
            print(f"Error: Video file {video_path} not found!")
            return
            
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer if output path provided
        if output_path:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 30 == 0:  # Print progress every 30 frames
                print(f"Processing frame {frame_count}/{total_frames}")
            
            # Detect objects
            detections = self.detector.detect_objects(frame)
            
            # Draw detections
            annotated_frame = self.detector.draw_detections(frame, detections)
            
            # Write frame if output specified
            if output_path:
                out.write(annotated_frame)
            
            # Display frame (optional)
            cv2.imshow('YOLO Video Processing', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        processing_time = time.time() - start_time
        print(f"Processing completed in {processing_time:.2f} seconds")
        
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
    
    def detect_from_image(self, image_path, save_path=None):
        """
        Process single image
        """
        if not os.path.exists(image_path):
            print(f"Error: Image file {image_path} not found!")
            return
            
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not load image {image_path}")
            return
        
        # Detect objects
        detections = self.detector.detect_objects(frame)
        
        # Draw detections
        annotated_frame = self.detector.draw_detections(frame, detections)
        
        print(f"Detected {len(detections)} objects:")
        for detection in detections:
            x1, y1, x2, y2, conf, class_id = detection
            class_name = self.detector.class_names[class_id]
            print(f"  - {class_name}: {conf:.2f} at [{int(x1)},{int(y1)},{int(x2)},{int(y2)}]")
        
        # Display result
        cv2.imshow('YOLO Detection Result', annotated_frame)
        print("Press any key to close the image window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save if path provided
        if save_path:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, annotated_frame)
            print(f"Result saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='YOLO Object Detection System')
    parser.add_argument('--mode', type=str, default='webcam', 
                       choices=['webcam', 'image', 'video'], 
                       help='Detection mode: webcam, image, or video')
    parser.add_argument('--input', type=str, default='0',
                       help='Input source: camera ID, image path, or video path')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for processed image or video')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
                       help='YOLO model to use')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold (0.0 to 1.0)')
    
    args = parser.parse_args()
    
    # Initialize the detector
    print(f"Initializing YOLO detector with model: {args.model}")
    yolo_system = RealTimeYOLO(args.model, args.conf)
    
    # Process based on mode
    if args.mode == 'webcam':
        try:
            camera_id = int(args.input)
        except ValueError:
            camera_id = 0
        yolo_system.detect_from_webcam(camera_id)
    elif args.mode == 'image':
        yolo_system.detect_from_image(args.input, args.output)
    elif args.mode == 'video':
        yolo_system.detect_from_video(args.input, args.output)

if __name__ == "__main__":
    main()