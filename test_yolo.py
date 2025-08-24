import cv2
import numpy as np
import os

# Create test images directory
os.makedirs("test_images", exist_ok=True)

# Create some simple test images
def create_test_images():
    # Create a street scene
    street_img = np.ones((400, 600, 3), dtype=np.uint8) * 200  # Light gray background
    
    # Add road
    cv2.rectangle(street_img, (0, 300), (600, 400), (100, 100, 100), -1)
    
    # Add cars
    cv2.rectangle(street_img, (100, 320), (250, 370), (0, 0, 255), -1)  # Red car
    cv2.rectangle(street_img, (350, 310), (500, 360), (255, 0, 0), -1)  # Blue car
    
    # Add person
    cv2.rectangle(street_img, (300, 250), (320, 320), (0, 255, 0), -1)  # Person
    
    cv2.imwrite("test_images/street.jpg", street_img)
    
    # Create an indoor scene
    indoor_img = np.ones((400, 600, 3), dtype=np.uint8) * 240  # Very light gray background
    
    # Add furniture
    cv2.rectangle(indoor_img, (100, 250), (300, 380), (139, 69, 19), -1)  # Table
    cv2.rectangle(indoor_img, (150, 200), (250, 250), (50, 50, 50), -1)   # Laptop
    cv2.rectangle(indoor_img, (400, 280), (450, 380), (139, 69, 19), -1)  # Chair
    
    # Add person
    cv2.rectangle(indoor_img, (350, 200), (370, 280), (0, 255, 0), -1)  # Person
    
    cv2.imwrite("test_images/indoor.jpg", indoor_img)
    
    print("Test images created in test_images/ directory")

if __name__ == "__main__":
    create_test_images()
    print("Run the following commands to test:")
    print("1. For webcam: python yolo_detector.py --mode webcam")
    print("2. For test images: python yolo_detector.py --mode image --input test_images/street.jpg")
    print("3. For test images with output: python yolo_detector.py --mode image --input test_images/street.jpg --output results/street_result.jpg")