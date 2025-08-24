import os
import glob
from yolo_detector import RealTimeYOLO

def batch_process_images(input_folder, output_folder, model_path='yolov8n.pt', conf_threshold=0.5):
    """
    Process all images in a folder
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    
    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, extension)))
    
    if not image_files:
        print(f"No images found in {input_folder}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Initialize detector
    detector = RealTimeYOLO(model_path, conf_threshold)
    
    # Process each image
    for i, image_path in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        
        # Generate output path
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_folder, f"{name}_detected{ext}")
        
        # Process image
        detector.detect_from_image(image_path, output_path)
    
    print(f"Batch processing completed. Results saved to {output_folder}")

if __name__ == "__main__":
    # Create test images if they don't exist
    if not os.path.exists("test_images"):
        import test_yolo
        test_yolo.create_test_images()
    
    # Process all test images
    batch_process_images("test_images", "results", "yolov8n.pt", 0.5)