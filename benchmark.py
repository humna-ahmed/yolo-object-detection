from yolo_detector import RealTimeYOLO
import time
import cv2
import os

def benchmark_models():
    """
    Compare performance of different YOLO models
    """
    models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt']
    test_image = 'test_images/street.jpg'
    
    if not os.path.exists(test_image):
        print("Test image not found. Run test_yolo.py first.")
        return
    
    frame = cv2.imread(test_image)
    if frame is None:
        print("Could not load test image")
        return
    
    print("Benchmarking YOLO models (10 runs each):")
    print("-" * 50)
    
    for model in models:
        print(f"\nTesting {model}:")
        detector = RealTimeYOLO(model)
        
        # Warm up
        detector.detector.detect_objects(frame)
        
        # Benchmark
        start_time = time.time()
        for _ in range(10):
            detections = detector.detector.detect_objects(frame)
        
        avg_time = (time.time() - start_time) / 10
        print(f"  Average detection time: {avg_time:.3f}s")
        print(f"  Estimated FPS: {1/avg_time:.1f}")
        print(f"  Objects detected: {len(detections)}")

if __name__ == "__main__":
    benchmark_models()