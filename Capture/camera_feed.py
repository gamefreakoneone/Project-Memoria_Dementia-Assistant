from ultralytics import YOLO
import cv2
from datetime import datetime
import os
import time
from audio_capture import AudioRecorder

def save_last_frame_screenshot(video_path, screenshot_dir):
    """Extract and save the last frame from a video file as a screenshot."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file for screenshot: {video_path}")
            return None
        
        # Get total frame count and go to last frame
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None:
            # Generate screenshot filename based on video filename
            video_basename = os.path.splitext(os.path.basename(video_path))[0]
            screenshot_filename = os.path.join(screenshot_dir, f"{video_basename}.jpg")
            cv2.imwrite(screenshot_filename, frame)
            print(f"Saved screenshot: {screenshot_filename}")
            return screenshot_filename
        else:
            print(f"Error: Could not read last frame from video: {video_path}")
            return None
    except Exception as e:
        print(f"Error saving screenshot: {e}")
        return None

def camera_feed():
    # Get the project root directory (parent of Capture folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Load YOLO11 model
    model = YOLO("yolo11n.pt")
    
    # Detect all available cameras
    camera_indices = [1]
    cameras = []
    video_writers = {}  # Track video writers for each camera
    audio_recorders = {}  # Track audio recorders for each camera
    recording_active = {}  # Track recording state for each camera
    video_output_dirs = {}  # Track video output directories per camera
    audio_output_dirs = {}  # Track audio output directories per camera
    screenshot_output_dirs = {}  # Track screenshot output directories per camera
    
    # Buffer tracking for detection grace period
    last_person_detected_time = {}  # Track when person was last detected per camera
    current_video_filename = {}  # Track current video filename for screenshot
    DETECTION_BUFFER_SECONDS = 2  # 2-second buffer before stopping recording

    for idx in camera_indices:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Try higher resolution
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        if cap.isOpened():
            cameras.append((idx, cap))
            video_writers[idx] = None
            audio_recorders[idx] = AudioRecorder()
            recording_active[idx] = False
            last_person_detected_time[idx] = None
            current_video_filename[idx] = None
            # Create camera-specific output directories
            video_output_dirs[idx] = os.path.join(project_root, "Storage", "video_recordings", f"camera_{idx}")
            audio_output_dirs[idx] = os.path.join(project_root, "Storage", "audio_recordings", f"camera_{idx}")
            screenshot_output_dirs[idx] = os.path.join(project_root, "Storage", "screenshots", f"camera_{idx}")
            os.makedirs(video_output_dirs[idx], exist_ok=True)
            os.makedirs(audio_output_dirs[idx], exist_ok=True)
            os.makedirs(screenshot_output_dirs[idx], exist_ok=True)
            print(f"Camera {idx} opened successfully!")
        else:
            print(f"Camera {idx} not available")
    
    if not cameras:
        print("Error: No cameras available")
        return
    
    print("Press 'q' to quit")
    
    # COCO class ID for 'person' is 0
    PERSON_CLASS_ID = 0

    frame_width = int(cameras[0][1].get(cv2.CAP_PROP_FRAME_WIDTH)) 
    frame_height = int(cameras[0][1].get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 20 
    
    while True:
        # Process each camera
        for idx, cap in cameras:
            ret, frame = cap.read() # here ret stands for return value and frame stands for the frame read from the camera
            
            if not ret:
                print(f"Error: Could not read frame from camera {idx}")
                continue
            
            # Run YOLO11 inference on the frame
            results = model(frame, verbose=False)
            
            # Process detections
            person_detected = False
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        if class_id == PERSON_CLASS_ID:
                            person_detected = True
                            break
                
                current_time = time.time()
                
                # Update last detection time if person is detected
                if person_detected:
                    last_person_detected_time[idx] = current_time
                    
                if person_detected and not recording_active[idx]:
                    # Start new recording
                    # Remmeber to store the timestamp as an object or string?? 
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                    video_filename = os.path.join(video_output_dirs[idx], f"camera_{idx}_{timestamp}.mp4")
                    audio_filename = os.path.join(audio_output_dirs[idx], f"camera_{idx}_{timestamp}.mp3")
                    current_video_filename[idx] = video_filename
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
                    video_writers[idx] = cv2.VideoWriter(video_filename, fourcc, fps, 
                                                     (frame_width, frame_height))
                    # Start audio recording
                    audio_recorders[idx].start_recording(audio_filename)
                    recording_active[idx] = True
                    print(f"Started recording: {video_filename}")
                elif not person_detected and recording_active[idx]:
                    # Check if buffer time has elapsed since last detection
                    time_since_last_detection = current_time - last_person_detected_time[idx]
                    
                    if time_since_last_detection >= DETECTION_BUFFER_SECONDS:
                        # Buffer expired, stop recording
                        video_writers[idx].release()
                        video_writers[idx] = None
                        # Stop audio recording
                        audio_recorders[idx].stop_recording()
                        recording_active[idx] = False
                        print(f"Stopped recording for camera {idx} (no detection for {DETECTION_BUFFER_SECONDS}s)")
                        
                        # Save last frame as screenshot
                        if current_video_filename[idx]:
                            save_last_frame_screenshot(current_video_filename[idx], screenshot_output_dirs[idx])
                            current_video_filename[idx] = None
                        
                        # Here we should pass the video path, audio_file path, screenshot path to the consolidator agent which will then call the subagents
                    # else: continue recording during buffer period

                if recording_active[idx] and video_writers[idx] is not None:
                    video_writers[idx].write(frame)


                for box in boxes: # Honestly this is more for reference
                    class_id = int(box.cls[0])
                    
                    if class_id == PERSON_CLASS_ID:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        
                        # Draw bounding box (green color)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw label with confidence
                        label = f"Human: {confidence:.2f}"
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        
                        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                      (x1 + label_size[0], y1), (0, 255, 0), -1)
                        
                        cv2.putText(frame, label, (x1, y1 - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Count humans detected
            human_count = sum(1 for result in results for box in result.boxes 
                             if int(box.cls[0]) == PERSON_CLASS_ID)
            
            # Display human count on frame
            count_text = f"Humans Detected: {human_count}"
            cv2.putText(frame, count_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if recording_active[idx]:
                cv2.putText(frame, "Recording", (10, 60), # Here (10, 60) is the position of the text
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 135), 2)
            
            # Display the frame with camera index in window name
            cv2.imshow(f"Camera {idx} - Human Detection", frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup - release all cameras and stop any active recordings
    for idx, cap in cameras:
        cap.release()
        if video_writers[idx] is not None:
            video_writers[idx].release()
        if recording_active.get(idx, False):
            audio_recorders[idx].stop_recording()
            # Save last frame as screenshot for any active recordings
            if current_video_filename.get(idx):
                save_last_frame_screenshot(current_video_filename[idx], screenshot_output_dirs[idx])
    cv2.destroyAllWindows()
    print("All cameras released and windows closed")


if __name__ == "__main__":
    camera_feed()