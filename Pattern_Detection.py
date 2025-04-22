from ultralytics import YOLO
import torch
import cv2
import numpy as np
import time

def yolov8_video_inference(model_path, video_source=0, image_size=640, conf_threshold=0.25, iou_threshold=0.45):
    """
    Perform object detection inference using YOLOv8 model on a live video stream.
    
    Args:
        model_path (str): Path to the YOLOv8 model file.
        video_source (int or str): Camera index (0 for default webcam) or video file path.
        image_size (int): Size of the image for inference.
        conf_threshold (float): Confidence threshold for detections.
        iou_threshold (float): IoU threshold for NMS.
    """
    # Load the YOLOv8 model
    model = YOLO(model_path)
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_source)
    
    # Check if the video source was opened successfully
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source {video_source}")
    
    # Get video properties for FPS calculation
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:  # For webcams that don't report FPS
        fps = 30
    
    # Initialize variables for FPS calculation
    frame_count = 0
    start_time = time.time()
    fps_display = 0
    
    while True:
        # Read a frame from the video source
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or error reading frame.")
            break
        
        # Run inference on the frame
        results = model.predict(
            source=frame,
            imgsz=image_size,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        # Get the rendered frame with detections drawn on it
        rendered_frame = results[0].plot()
        
        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0:  # Update FPS every second
            fps_display = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
        
        # Add FPS information to the frame
        cv2.putText(
            rendered_frame, 
            f"FPS: {fps_display:.2f}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )
        
        # Display the frame with detections
        cv2.imshow("YOLOv8 Live Detection", rendered_frame)
        
        # Check for user input to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    model_path = 'model.pt'  # Replace with your model path
    
    # For webcam (default camera)
    # yolov8_video_inference(model_path)
    
    # For a specific camera index
    yolov8_video_inference(model_path, video_source=1)
    
    # For a video file
    # yolov8_video_inference(model_path, video_source='path/to/video.mp4')